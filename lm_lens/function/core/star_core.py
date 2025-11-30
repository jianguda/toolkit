from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from lm_lens.function.utils.attribution import Attr
from lm_lens.function.config import Configure
from lm_lens.function.utils.loader import NeoLoader, Metric
from lm_lens.function.config.shared import DEVICE, MAX_FOCUSING_NUM
from lm_lens.function.utils import get_attr, compute_sss, freeze_params, constrained_lstsq, obtain_loss
from lm_lens.function.utils.optimizer import Seme


class StarCore:
    def __init__(self, cfg: Configure):
        self.cfg = cfg
        self.tokenizer = NeoLoader.load_tokenizer(self.cfg.model_name)

        self.config, self.model, self.attrs = NeoLoader.load_model(self.cfg.model_name)
        self.model.to(DEVICE)
        self.attr = Attr(self.cfg, self.model, self.tokenizer, self.attrs)

        # ... for self._reinit()
        self.inplace_weights = {
            name: parameters
            for name, parameters in self.model.named_parameters()
            if self.attrs['ff_output'] + '.weight' in name
        }
        self.backup_weights = {k: v.detach().clone() for k, v in self.inplace_weights.items()}

        self.embeddings_matrix = self.get_embedding_matrix()
        self.lm_head_matrix = self.get_lm_head_matrix()

    def _reinit(self):
        with torch.no_grad():
            for k, v in self.inplace_weights.items():
                v[...] = self.backup_weights[k]

    def generate(self, prompt, target):
        oracle_tokens = []
        # logger.debug(f'{target=}')
        target_labels = [label for label in self.tokenizer.encode(target, add_special_tokens=False)]
        if 'codegen' in self.cfg.model_name:
            target_labels = [self.tokenizer.encode(' ')[0] if label > 50256 else label for label in target_labels]
        target_tokens = [self.tokenizer.decode(label, skip_special_tokens=True) for label in target_labels]
        # logger.warning(f'{target_labels=}')
        # logger.warning(f'{target_tokens=}')

        argmax_labels = []
        for target_token, target_label in zip(target_tokens, target_labels):
            probing_prompt = prompt + ''.join(oracle_tokens)
            oracle_tokens.append(target_token)
            logits = self._predict(probing_prompt)
            # logger.critical(f'{target_token=}')
            argmax_label = torch.argmax(logits, dim=-1).item()
            argmax_labels.append(argmax_label)
        # logger.warning(f'{target_labels=}')
        # logger.warning(f'{argmax_labels=}')

        if 'codegen' in self.cfg.model_name:
            argmax_labels = [self.tokenizer.encode(' ')[0] if label > 50256 else label for label in argmax_labels]
        argmax_tokens = [self.tokenizer.decode(label, skip_special_tokens=True) for label in argmax_labels]
        # logger.info(f'{target_tokens=}')
        # logger.info(f'{argmax_tokens=}')
        return target_tokens, argmax_tokens, target_labels, argmax_labels

    def _predict(self, prompt):
        if "CodeQwen" in self.cfg.model_name or "Qwen2.5-Coder" in self.cfg.model_name:
            inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(DEVICE)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        logits = F.softmax(outputs.scores[0], dim=-1)
        return logits

    def _project(self, logits, watch_labels=None):
        sorted_probs, sorted_labels = logits.sort(dim=-1, descending=True)
        argmax_probs = sorted_probs.cpu().numpy()[0].tolist()
        argmax_labels = sorted_labels.cpu().numpy()[0].tolist()

        # # filter sorted_indices using the domain vocabs
        # if self.allow_vocab_labels is not None:
        #     argmax_vocab_probs = list()
        #     argmax_vocab_labels = list()
        #     for argmax_prob, argmax_label in zip(argmax_probs, argmax_labels):
        #         if argmax_label in self.allow_vocab_labels:
        #             argmax_vocab_probs.append(argmax_prob)
        #             argmax_vocab_labels.append(argmax_label)
        #     argmax_probs = argmax_vocab_probs
        #     argmax_labels = argmax_vocab_labels
        # if self.block_vocab_labels is not None:
        #     argmax_vocab_probs = list()
        #     argmax_vocab_labels = list()
        #     for argmax_prob, argmax_label in zip(argmax_probs, argmax_labels):
        #         if argmax_label not in self.block_vocab_labels:
        #             argmax_vocab_probs.append(argmax_prob)
        #             argmax_vocab_labels.append(argmax_label)
        #     argmax_probs = argmax_vocab_probs
        #     argmax_labels = argmax_vocab_labels

        watch_probs = []
        watch_ranks = []
        # logger.warning(f'{watch_labels=}')
        if watch_labels is not None:
            for watch_label in watch_labels:
                # watch_token = self.tokenizer.decode(watch_label, skip_special_tokens=True)
                # logger.critical(f'{watch_label=}, {watch_token=}')
                # logger.critical(f'{torch.tensor(argmax_labels) == watch_label=}')
                ranking = torch.nonzero(torch.tensor(argmax_labels) == watch_label).flatten()
                if 'codegen' in self.cfg.model_name:
                    if watch_label == 50256:
                        ranking = torch.nonzero(torch.tensor(argmax_labels) >= watch_label)[0].flatten()
                # logger.warning(f'{argmax_labels=}')
                # logger.debug(f'{ranking=}')
                prob = argmax_probs[ranking]
                watch_probs.append(prob)
                watch_ranks.append(int(ranking))

        # we only care about top tokens
        argmax_probs = argmax_probs[:MAX_FOCUSING_NUM]
        argmax_labels = argmax_labels[:MAX_FOCUSING_NUM]
        return argmax_labels, argmax_probs, watch_probs, watch_ranks

    # a simple implementation, following this one...
    def do_benchmark(self, nested_triplets: list[list[tuple]]):
        # # xxx
        # block_vocab_units = ['\n', '\r', '\t', '\n\n']  # + [self.tokenizer.eos_token]
        # unique_vocab_labels = set([label for unit in block_vocab_units for label in self.tokenizer.encode(unit, add_special_tokens=False)])
        # self.block_vocab_labels = sorted(list(unique_vocab_labels))
        # for statistics
        pre_deltas_g = list()
        pre_deltas_s = list()
        post_deltas_g = list()
        post_deltas_s = list()
        # index for generalization and specificity
        range_g = len(nested_triplets)  # for index_g in range(range_g), index_g stays unchanged in iteration...
        range_s = len(nested_triplets[0])  # for index_s in range(range_s), index_s stays unchanged in iteration...

        def _utility(_triplet, _target_label):
            _prompt, _argmax_symbol, _target_symbols = _triplet
            _logits = self._predict(_prompt)
            _, _top_probs, _watch_probs, _watch_ranks = self._project(_logits, [_target_label])

            _argmax_prob = _top_probs[0]
            _target_prob = _watch_probs[0]
            _target_rank = _watch_ranks[0]

            # ...
            _prob_delta = abs(_target_prob - _argmax_prob)
            _rank_delta = abs(_target_rank - 0)
            _is_matched = 1 if _target_rank == 0 else 0
            return _prob_delta, _rank_delta, _is_matched

        for index_g in range(range_g):
            for index_s in range(range_s):
                triplet = nested_triplets[index_g][index_s]
                prompt, argmax_symbol, target_symbols = triplet

                for target_symbol in target_symbols:
                    # logger.critical(f'{argmax_symbol=}')
                    # logger.critical(f'{target_symbol=}')
                    # xxx
                    argmax_labels = [label for label in self.tokenizer.encode(argmax_symbol, add_special_tokens=False)]
                    target_labels = [label for label in self.tokenizer.encode(target_symbol, add_special_tokens=False)]
                    # take the first different label-pair for study
                    argmax_label = None
                    target_label = None
                    for a_label, t_label in zip(argmax_labels, target_labels):
                        if a_label != t_label:
                            argmax_label = a_label
                            target_label = t_label
                            break

                    # reinitialize
                    self._reinit()
                    # TODO in some cases where the expected argmax is not the actual argmax
                    # thereby we edit the model to guarantee our assumption
                    # and further, start the normal process on benchmark
                    logits = self._predict(prompt)
                    top_labels, _, _, _ = self._project(logits)
                    # this is a constant default setting
                    cfg_backup = deepcopy(self.cfg)
                    self.cfg = Configure(self.cfg.scale, self.cfg.data_name, self.cfg.model_name)
                    self.sgd_process(prompt, target_label=argmax_label, argmax_label=top_labels[0])
                    self.cfg = cfg_backup

                    # before editing
                    # generalization
                    for _index_s in range(range_s):
                        if _index_s == index_s:
                            continue
                        _triplet = nested_triplets[index_g][_index_s]
                        pre_deltas_g.append(_utility(_triplet, target_label))
                    # specificity
                    for _index_g in range(range_g):
                        if _index_g == index_g:
                            continue
                        _triplet = nested_triplets[_index_g][index_s]
                        pre_deltas_s.append(_utility(_triplet, target_label))

                    # conduct editing
                    logits = self._predict(prompt)
                    top_labels, _, _, _ = self._project(logits)
                    # this is a variable setting (affected by ablation study)
                    self.process(prompt, target_label=target_label, argmax_label=top_labels[0])

                    # after editing
                    # generalization
                    for _index_s in range(range_s):
                        if _index_s == index_s:
                            continue
                        _triplet = nested_triplets[index_g][_index_s]
                        post_deltas_g.append(_utility(_triplet, target_label))
                    # specificity
                    for _index_g in range(range_g):
                        if _index_g == index_g:
                            continue
                        _triplet = nested_triplets[_index_g][index_s]
                        post_deltas_s.append(_utility(_triplet, target_label))
        return pre_deltas_g, pre_deltas_s, post_deltas_g, post_deltas_s

    # always use SGD to guarantee the initial situation in doing benchmark
    def sgd_process(self, prompt, target_label, argmax_label, epoch_num=10):
        # we skip all correct predictions, and focus on these most needed positions
        if target_label == argmax_label:
            if 'codegen' in self.cfg.model_name:
                target_label = self.tokenizer.encode(' ')[0] if target_label > 50256 else target_label
                argmax_label = self.tokenizer.encode(' ')[0] if argmax_label > 50256 else argmax_label
            target_token = self.tokenizer.decode(target_label, skip_special_tokens=True)
            argmax_token = self.tokenizer.decode(argmax_label, skip_special_tokens=True)
            logger.info(f'{target_token=} ({target_label}), {argmax_token=} ({argmax_label})')
            logger.warning(f'skip editing: the target_label is already ranked #0')

        prompts = [prompt]
        encoding = self.tokenizer(prompts, add_special_tokens=False, truncation=True, return_tensors="pt")
        target_labels = [target_label]
        argmax_labels = [argmax_label]
        input_ids = encoding["input_ids"].to(DEVICE)
        # logger.debug(f'{input_ids.shape=}')
        ground_truth = torch.tensor(target_labels).to(DEVICE)
        # logger.debug(f'{ground_truth.shape=}')

        param_groups = freeze_params(self.model, self.attrs)
        optimizer = optim.SGD(param_groups, lr=1e-2)
        self.model.train()
        for step in range(epoch_num):
            loss = obtain_loss(self.model, input_ids, ground_truth, is_batch=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def process(self, prompt, target_label, argmax_label, epoch_num=10):
        # we skip all correct predictions, and focus on these most needed positions
        if target_label == argmax_label:
            if 'codegen' in self.cfg.model_name:
                target_label = self.tokenizer.encode(' ')[0] if target_label > 50256 else target_label
                argmax_label = self.tokenizer.encode(' ')[0] if argmax_label > 50256 else argmax_label
            target_token = self.tokenizer.decode(target_label, skip_special_tokens=True)
            argmax_token = self.tokenizer.decode(argmax_label, skip_special_tokens=True)
            logger.info(f'{target_token=} ({target_label}), {argmax_token=} ({argmax_label})')
            logger.warning(f'skip editing: the target_label is already ranked #0')

        prompts = [prompt]
        encoding = self.tokenizer(prompts, add_special_tokens=False, truncation=True, return_tensors="pt")
        target_labels = [target_label]
        argmax_labels = [argmax_label]
        input_ids = encoding["input_ids"].to(DEVICE)
        # logger.debug(f'{input_ids.shape=}')
        ground_truth = torch.tensor(target_labels).to(DEVICE)
        # logger.debug(f'{ground_truth.shape=}')

        if self.cfg.ABL0_APPROACH == 'finetune':
            param_groups = freeze_params(self.model, self.attrs)
            optimizer = optim.SGD(param_groups, lr=1e-2 * self.cfg.scale)
            self.model.train()
            for step in range(epoch_num):
                loss = obtain_loss(self.model, input_ids, ground_truth, is_batch=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            scaled_semantic_steers, nonzero_rows = self.obtain_sss(input_ids, target_labels, argmax_labels)
            trainable_layers = self.specify_trainable_layers(input_ids, ground_truth, amount=self.cfg.ABL1_ABLATION_LAYER)
            param_groups = freeze_params(self.model, self.attrs, trainable_layers)
            optimizer = Seme(param_groups, lr=1e-2 * self.cfg.scale)
            self.model.train()
            for step in range(epoch_num):
                self.do_repair(optimizer, input_ids, ground_truth, scaled_semantic_steers, nonzero_rows, is_batch=False, trainable_layers=trainable_layers)

    def get_embedding_matrix(self):
        embeddings_matrix = get_attr(self.model, self.attrs['embedding'])
        embeddings_matrix = embeddings_matrix.weight.detach()  # .cpu()
        embeddings_matrix = embeddings_matrix.to(torch.float)
        return embeddings_matrix

    def get_lm_head_matrix(self):
        lm_head_matrix = get_attr(self.model, self.attrs['lm_head'])
        lm_head_matrix = lm_head_matrix.weight.detach()  # .cpu()
        lm_head_matrix = lm_head_matrix.to(torch.float)
        return torch.linalg.pinv(lm_head_matrix.T)

    # obtain the output of each MLP hidden layer, denote as Z[layer_idx]
    def obtain_layer_acts(self, input_ids, labels, is_batch=True):
        # logger.debug(f'{labels.shape=}')

        # layer_act_output.shape=torch.Size([1, 4096])
        layer_act_outputs = dict()

        def act_fn_hook(layer_idx):
            def hook(module, input, output):
                layer_act_outputs[layer_idx] = output[:, -labels.size(-1):, :].flatten(0, 1)  # [N, 1, 4096] => [1*N, 4096]
            return hook

        hooks = list()
        model_layers = get_attr(self.model, self.attrs['layers'])
        num_layers = len(model_layers)
        for layer_idx in range(num_layers):
            model_layer = model_layers[layer_idx]
            act_fn = get_attr(model_layer, self.attrs['ff_act'])
            hook = act_fn.register_forward_hook(act_fn_hook(layer_idx))
            hooks.append(hook)

        loss = obtain_loss(self.model, input_ids, labels, is_batch)

        # detach the hooks
        for hook in hooks:
            hook.remove()

        return layer_act_outputs, loss

    def obtain_sss(self, input_ids, target_labels, argmax_labels):
        scaled_semantic_steers = list()
        for target_label, argmax_label in zip(target_labels, argmax_labels):
            scaled_semantic_steer = compute_sss(self.lm_head_matrix, target_label, argmax_label)
            scaled_semantic_steers.append(scaled_semantic_steer)
        scaled_semantic_steers = torch.cat(scaled_semantic_steers, dim=0)

        nonzero_rows = defaultdict(list)
        # w/ firing (no longer in need...)
        if self.cfg.ABL2_ABLATION_WHICH in ['autoly', 'evenly', 'layerly']:
            # for constrained layer_deltas
            located_neurons = self.attr.get_layer_filters(input_ids, target_labels)
            for layer_idx, neuron_idx in located_neurons:
                nonzero_rows[layer_idx].append(neuron_idx)

        return scaled_semantic_steers, nonzero_rows

    def obtain_layer_deltas(self, layer_act_outputs, scaled_semantic_steers, nonzero_rows):
        layer_deltas = dict()
        for layer_idx, layer_act_output in layer_act_outputs.items():
            # make sure W is float (to be used in optimizer), so Z and S shall be float
            # Z.shape = torch.Size([N, 4096])
            Z = layer_act_output.to(torch.float)
            # S.shape = torch.Size([N, 1024])
            S = scaled_semantic_steers.to(torch.float)

            # W.shape=torch.Size([4096, 1024])
            if self.cfg.ABL2_ABLATION_WHICH in ['autoly', 'evenly', 'layerly']:
                # compute the deltaW (w/ constraints: specifying some rows as zeros)
                W = constrained_lstsq(Z, S, nonzero_rows[layer_idx])
            else:
                # step3: compute the deltaW (w/o constraints)
                W = torch.linalg.lstsq(Z, S).solution
            # logger.debug(f'{W.shape=}')
            # logger.debug(f'{layer_idx=}, {W=}')

            # # TODO here we mainly focus on the magnitude, not the sign, since we only utilize the abs-values
            # # however, when doing attribution scores, we can also focus on the magnitude (but the explanation may be a bit more complex...)
            # if self.cfg.ABL2_ABLATION_WHICH == 'maskmax':
            #     ratio_quantile = self.cfg.ABL2_ABLATION_RATIO
            #     abs_W = torch.abs(W.detach())
            #     # Compute the 10th percentile (1/10 minimal elements)
            #     threshold = np.quantile(abs_W.cpu(), ratio_quantile)
            #     # Set elements below the threshold to zero
            #     W = torch.where(abs_W < threshold, W, 0.0)
            # elif self.cfg.ABL2_ABLATION_WHICH == 'maskmin':  # this one is good
            #     ratio_quantile = 1 - self.cfg.ABL2_ABLATION_RATIO
            #     abs_W = torch.abs(W.detach())
            #     # Compute the 10th percentile (1/10 minimal elements)
            #     threshold = np.quantile(abs_W.cpu(), ratio_quantile)
            #     # Set elements below the threshold to zero
            #     W = torch.where(abs_W > threshold, W, 0.0)
            # elif self.cfg.ABL2_ABLATION_WHICH == 'maskrand':
            #     ratio_quantile = self.cfg.ABL2_ABLATION_RATIO
            #     probabilities = torch.rand(W.shape)
            #     # Set elements below the threshold to zero
            #     W = torch.where(probabilities < ratio_quantile, W, 0.0)
            # else:
            #     pass

            # ...
            layer_delta = W if 'gpt2' in self.cfg.model_name else W.T
            layer_deltas[layer_idx] = layer_delta

        return layer_deltas

    def specify_trainable_layers(self, input_ids, labels, amount='all'):
        if amount == 'all':
            return None

        model_layers = get_attr(self.model, self.attrs['layers'])
        num_layers = len(model_layers)

        def skip_layer(module, input, output):
            # pass input directly as output
            return input[0]

        layer_losses = list()
        for layer_idx in range(num_layers):
            model_layer = model_layers[layer_idx]
            mlp = get_attr(model_layer, 'mlp')
            hook_handle = mlp.register_forward_hook(skip_layer)
            loss = obtain_loss(self.model, input_ids, labels, is_batch=False)
            layer_losses.append(loss.item())
            hook_handle.remove()

        # corresponding to the case where the missing layer cause the largest loss...
        if amount == 'one':
            number = 1
        elif amount == '1quarter':
            number = num_layers * 1 // 4
        elif amount == '2quarter':
            number = num_layers * 2 // 4
        elif amount == '3quarter':
            number = num_layers * 3 // 4
        else:
            number = num_layers
        topn = sorted(range(num_layers), key=lambda i: layer_losses[i], reverse=True)[:number]
        return topn

    def do_repair(self, optimizer, input_ids, ground_truth, scaled_semantic_steers, nonzero_rows, is_batch, trainable_layers=None):
        layer_act_outputs, loss = self.obtain_layer_acts(input_ids, ground_truth, is_batch=is_batch)
        layer_deltas = self.obtain_layer_deltas(layer_act_outputs, scaled_semantic_steers, nonzero_rows)
        optimizer.update_delta(layer_deltas, trainable_layers)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    def pipeline(self, source: str, target: str, epoch_num=10, study_of=False):
        # we reinit model to avoid interferes
        self._reinit()
        self.model.eval()

        # refresh
        if study_of:
            pre_gen_tokens = list()
        else:
            target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(source, target)
            pre_gen_tokens = argmax_tokens

        # # here we try something new (but seems not that useful...)
        # from lm_lens.function.utils import apply_signed_grad_to_model
        # self.model = apply_signed_grad_to_model(self.model)

        if self.cfg.ABL0_APPROACH == 'me-iter' or self.cfg.ABL0_APPROACH == 'me-sgd':
            # ...iterative
            oracle_tokens = []
            num_iter = len(target_labels)

            self.model.train()
            for idx, (target_label, argmax_label) in enumerate(zip(target_labels, argmax_labels)):
                logger.success('=' * 9 + f'{idx + 1}/{num_iter}' + '=' * 9)
                target_token = self.tokenizer.decode(target_label, skip_special_tokens=True)
                # argmax_token = self.tokenizer.decode(argmax_label, skip_special_tokens=True)

                # we skip all correct predictions, and focus on these most needed positions
                if target_label == argmax_label:
                    # post_gen_tokens.append(argmax_token)
                    oracle_tokens.append(target_token)
                    continue

                prompt = source + ''.join(oracle_tokens)
                prompts = [prompt]
                encoding = self.tokenizer(prompts, add_special_tokens=False, truncation=True, return_tensors="pt")
                target_labels = [target_label]
                argmax_labels = [argmax_label]
                input_ids = encoding["input_ids"].to(DEVICE)
                # logger.debug(f'{input_ids.shape=}')
                ground_truth = torch.tensor(target_labels).to(DEVICE)
                # logger.debug(f'{ground_truth.shape=}')

                # SGD for single-mode (notice: solving for only one issue)
                if self.cfg.ABL0_APPROACH == 'me-sgd':
                    param_groups = freeze_params(self.model, self.attrs)
                    optimizer = optim.SGD(param_groups, lr=1e-2)

                    for step in range(epoch_num):
                        optimizer.zero_grad()
                        loss = obtain_loss(self.model, input_ids, ground_truth, is_batch=False)
                        loss.backward()
                        optimizer.step()
                elif self.cfg.ABL0_APPROACH == 'me-iter':
                    # ...
                    scaled_semantic_steers, nonzero_rows = self.obtain_sss(input_ids, target_labels, argmax_labels)
                    # ...
                    trainable_layers = self.specify_trainable_layers(input_ids, ground_truth, amount=self.cfg.ABL1_ABLATION_LAYER)
                    param_groups = freeze_params(self.model, self.attrs, trainable_layers)
                    optimizer = Seme(param_groups)

                    for step in range(epoch_num):
                        # if target_label == argmax_label:
                        #     break
                        self.do_repair(optimizer, input_ids, ground_truth, scaled_semantic_steers, nonzero_rows, is_batch=False, trainable_layers=trainable_layers)
                    # # update the argmax_label in each step... (as early stopping!!!)
                    # there is no need to do this, since the loss is always decreasing, unless the epoch_number is super large (currently it is 10)
                    # and an early-stop may also affect the performance somehow (to confirm when necessary) ...
                    # post_logits = self._predict(prompt)
                    # argmax_label = torch.argmax(post_logits, dim=-1)

                # argmax_token = self.tokenizer.decode(argmax_label, skip_special_tokens=True)
                # # update argmax_token
                # post_gen_tokens.append(argmax_token)
                oracle_tokens.append(target_token)
        elif self.cfg.ABL0_APPROACH == 'me-batch':
            # ...batch (optimize multiple data...)
            encoding = self.tokenizer([source], add_special_tokens=False, truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(DEVICE)
            # logger.debug(f'{input_ids.shape=}')
            ground_truth = torch.tensor(target_labels).to(DEVICE)
            # logger.debug(f'{ground_truth.shape=}')
            # ...
            scaled_semantic_steers, nonzero_rows = self.obtain_sss(input_ids, target_labels, argmax_labels)
            # ...
            # TODO in the case of w/o neuron-firing, we shall put the init outside the loop (in the init function)???
            # w/o firing
            param_groups = freeze_params(self.model, self.attrs)
            optimizer = Seme(param_groups)
            # optimizer = optim.SGD(param_groups)
            # assert isinstance(optimizer, Seme)

            self.model.train()
            for step in range(epoch_num):
                # logger.success('=' * 9 + f'{step}' + '=' * 9)
                # # update...
                # if target_labels == argmax_labels:
                #     break
                # there is no need to do this, since the loss is always decreasing, unless the epoch_number is super large (currently it is 10)
                # and an early-stop may also affect the performance somehow (to confirm when necessary) ...
                # repair...
                try:
                    self.do_repair(optimizer, input_ids, ground_truth, scaled_semantic_steers, nonzero_rows, is_batch=True, trainable_layers=None)
                except Exception as e:
                    print(e)
                    break
        else:
            raise NotImplementedError

        self.model.eval()
        if study_of:
            target_tokens = list()
            post_gen_tokens = list()
        else:
            target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(source, target)
            post_gen_tokens = argmax_tokens

        return pre_gen_tokens, post_gen_tokens, target_tokens

    def finetune(self, source: str, target: str, epoch_num=10, study_of=False):
        # we reinit model to avoid interferes
        self._reinit()
        self.model.eval()

        # refresh
        if study_of:
            pre_gen_tokens = list()
        else:
            target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(source, target)
            pre_gen_tokens = argmax_tokens

        # ......
        dataset = PairedDataset([source], [target], self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1)
        # # Evaluation
        # pre_labels = self.predict(dataloader)
        # pre_peft_acc = Metric.same_accuracy(pre_labels, target_labels)

        param_groups = freeze_params(self.model, self.attrs)
        # optimizer = optim.AdamW(param_groups)
        optimizer = optim.SGD(param_groups, lr=1e-2)

        self.model.train()
        for epoch_idx in range(epoch_num):
            # logger.success('=' * 9 + f'{epoch_idx}' + '=' * 9)
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                # Combine input and ground truth
                # logger.debug(f'{input_ids.shape=}')
                # logger.debug(f'{labels.shape=}')
                combined_ids = torch.cat([input_ids, labels], dim=-1)
                ground_truth = combined_ids.clone()
                # Set the labels for the input part to -100 (ignored in loss computation)
                ground_truth[:, :input_ids.size(1)] = -100

                # The shift operation happens automatically in the forward pass when given both input_ids and labels
                outputs = self.model(input_ids=combined_ids, labels=ground_truth, output_hidden_states=True,
                                     return_dict=True)
                loss = outputs.loss
                # logits = outputs.logits
                # hidden_states = outputs.hidden_states

                loss.backward()
                optimizer.step()

        # # Evaluation
        # post_labels = self.predict(dataloader)
        # post_peft_acc = Metric.same_accuracy(post_labels, target_labels)
        # logger.debug(f'{epoch_idx=}, {pre_peft_acc=}, {post_peft_acc=}')

        self.model.eval()
        if study_of:
            target_tokens = list()
            post_gen_tokens = list()
        else:
            target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(source, target)
            post_gen_tokens = argmax_tokens

        return pre_gen_tokens, post_gen_tokens, target_tokens


from torch.utils.data import DataLoader, Dataset


class PairedDataset(Dataset):
    def __init__(self, input_lines, output_lines, tokenizer, max_length=1024):
        self.input_lines = input_lines
        self.output_lines = output_lines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_lines)

    def __getitem__(self, idx):
        input_line = self.input_lines[idx]
        output_line = self.output_lines[idx]

        input_encoding = self.tokenizer(input_line, add_special_tokens=False, truncation=True, return_tensors="pt")
        input_ids = input_encoding["input_ids"].to(DEVICE)
        if len(input_ids.shape) == 2:
            input_ids = input_ids.squeeze(0)
        # logger.debug(f'{input_ids.shape=}')

        output_encoding = self.tokenizer(output_line, add_special_tokens=False, truncation=True, return_tensors="pt")
        output_ids = output_encoding["input_ids"].to(DEVICE)
        if len(output_ids.shape) == 2:
            output_ids = output_ids.squeeze(0)
        # logger.debug(f'{output_ids.shape=}')

        return {"input_ids": input_ids, "labels": output_ids}
