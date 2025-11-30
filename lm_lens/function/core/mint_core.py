from copy import deepcopy
import math
from collections import defaultdict, deque
from functools import partial, reduce
from queue import Queue, PriorityQueue
from typing import List, Tuple, Callable, Dict

import torch
from torch import nn, optim
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from lm_lens.function.utils.attribution import Attr
from lm_lens.function.config import Configure
from lm_lens.function.utils.loader import NeoLoader, Metric
from lm_lens.function.config.shared import (
    MAX_PATCHING_NUM, CFG_FOCUSED_LAYERS, DEVICE, MAX_FOCUSING_NUM
)
from lm_lens.function.utils import get_attr, format_score, obtain_loss, freeze_params


class MintCore:
    def __init__(
            self,
            cfg: Configure,
            allow_vocab_labels: List = None,
            block_vocab_labels: List = None,
    ):
        self.cfg = cfg
        self.tokenizer = NeoLoader.load_tokenizer(self.cfg.model_name)
        self.allow_vocab_labels = allow_vocab_labels
        self.block_vocab_labels = block_vocab_labels

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

        self.embeddings_matrix = self.get_embedding_matrix().to('cpu')
        self.lm_head_matrix = self.get_lm_head_matrix().to('cpu')

        # 历史 traction，用于之后做正交投影
        # key: layer_idx, value: list[unit Tensor]（已经归一化的一维向量）
        self.history_tractions: Dict[int, List[Tensor]] = defaultdict(list)

    def _reinit(self):
        with torch.no_grad():
            for k, v in self.inplace_weights.items():
                v[...] = self.backup_weights[k]
        # 清空历史 traction，避免跨实验污染
        self.history_tractions.clear()

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
            # TODO using oracle_tokens to compose prompt for a fair evaluation
            probing_prompt = prompt + ''.join(oracle_tokens)
            oracle_tokens.append(target_token)
            logits = self._predict(probing_prompt)
            # logger.critical(f'{target_token=}')
            top_labels, top_probs, _target_probs, _target_ranks = self._project(logits, [target_label])
            argmax_label = top_labels[0]
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

        # filter sorted_indices using the domain vocabs
        if self.allow_vocab_labels is not None:
            argmax_vocab_probs = list()
            argmax_vocab_labels = list()
            for argmax_prob, argmax_label in zip(argmax_probs, argmax_labels):
                if argmax_label in self.allow_vocab_labels:
                    argmax_vocab_probs.append(argmax_prob)
                    argmax_vocab_labels.append(argmax_label)
            argmax_probs = argmax_vocab_probs
            argmax_labels = argmax_vocab_labels
        if self.block_vocab_labels is not None:
            argmax_vocab_probs = list()
            argmax_vocab_labels = list()
            for argmax_prob, argmax_label in zip(argmax_probs, argmax_labels):
                if argmax_label not in self.block_vocab_labels:
                    argmax_vocab_probs.append(argmax_prob)
                    argmax_vocab_labels.append(argmax_label)
            argmax_probs = argmax_vocab_probs
            argmax_labels = argmax_vocab_labels

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

    def get_embedding_matrix(self):
        embeddings_matrix = get_attr(self.model, self.attrs['embedding'])
        embeddings_matrix = embeddings_matrix.weight.detach()  # .cpu()
        embeddings_matrix = embeddings_matrix.to(torch.float).to('cpu')
        return embeddings_matrix

    def get_lm_head_matrix(self):
        lm_head_matrix = get_attr(self.model, self.attrs['lm_head'])
        lm_head_matrix = lm_head_matrix.weight.detach()  # .cpu()
        lm_head_matrix = lm_head_matrix.to(torch.float).to('cpu')
        return torch.linalg.pinv(lm_head_matrix.T)

    def _register_history_tractions(self, layer_tractions: Dict[int, Tensor]):
        """
        把本次 update 用到的 traction 存进历史，用于之后 token 做正交化。
        统一存成 CPU 上的一维单位向量，减少显存占用。
        """
        for layer_idx, v in layer_tractions.items():
            v_flat = v.detach().to('cpu').flatten()
            norm = v_flat.norm()
            if norm > 1e-8:
                self.history_tractions[layer_idx].append(v_flat / norm)

    def _project_onto_history_orthogonal(self, layer_idx: int, traction: Tensor) -> Tensor:
        """
        把 traction 投影到历史方向张成子空间的正交补上：
            u = v - sum_i (v·h_i) h_i
        其中 h_i 是历史的单位向量。
        最后保持与原 traction 近似相同的 scale，仅改变方向。
        """
        history = self.history_tractions.get(layer_idx)
        if not history:
            return traction

        u = traction.flatten()
        for h_unit in history:
            h_unit = h_unit.to(u.device)
            # h_unit 已经归一化，可以直接 dot
            coeff = torch.dot(u, h_unit)
            u = u - coeff * h_unit

        norm = u.norm()
        if norm < 1e-8:
            # 极端情况：完全落在历史子空间里，退回原始方向，避免数值异常
            return traction

        # 保持整体范数不变，只调整方向
        target_norm = traction.flatten().norm()
        if target_norm > 0:
            u = u / norm * target_norm
        else:
            u = u / norm

        return u.view_as(traction)

    # the simplified (but equivalent) implementation
    def compute_traction_simplified(self, target_label, argmax_label):
        output_argmax_embed = self.lm_head_matrix[argmax_label].detach().to(DEVICE)
        output_target_embed = self.lm_head_matrix[target_label].detach().to(DEVICE)
        # this implementation is merely for the case of semantic delta
        # medium_label is needed to implement the case of semantic basis
        semantic_drift = (output_target_embed - output_argmax_embed).flatten()

        # tune to weights range (varys by each layer)
        model_layers = get_attr(self.model, self.attrs['layers'])
        num_layers = len(model_layers)
        if CFG_FOCUSED_LAYERS == 'all':
            skipped_num_layers = int(num_layers * 0)
        elif CFG_FOCUSED_LAYERS == 'half':
            skipped_num_layers = int(num_layers * 0.5)
        elif CFG_FOCUSED_LAYERS == 'forth':
            skipped_num_layers = int(num_layers * 0.75)
        elif CFG_FOCUSED_LAYERS == 'last3':
            skipped_num_layers = int(num_layers - 3)
        else:
            raise NotImplementedError

        layer_tractions = dict()
        for layer_idx in range(skipped_num_layers, num_layers):
            # actually this step can be further omitted
            layer_drift = semantic_drift * ((layer_idx + 1) / num_layers)
            layer_drift_range = (torch.max(layer_drift) - torch.min(layer_drift)).item()
            if layer_drift_range == 0.0:
                # 极少数情况下全常数，跳过该层
                continue
            scaled_layer_drift = layer_drift / layer_drift_range

            # 投影到旧update方向的正交空间
            projected = self._project_onto_history_orthogonal(layer_idx, scaled_layer_drift)
            layer_tractions[layer_idx] = projected

        return layer_tractions

    # the vanilla implementation
    def compute_traction(self, target_label, argmax_label, medium_label=None):
        assert medium_label is not None
        # semantic bases at input side
        input_medium_base = self.embeddings_matrix[medium_label].detach().to(DEVICE)
        # semantic bases at output side
        output_argmax_base = self.lm_head_matrix[argmax_label].detach().to(DEVICE)
        output_target_base = self.lm_head_matrix[target_label].detach().to(DEVICE)

        # tune to weights range (varys by each layer)
        model_layers = get_attr(self.model, self.attrs['layers'])
        num_layers = len(model_layers)
        if CFG_FOCUSED_LAYERS == 'all':
            skipped_num_layers = int(num_layers * 0)
        elif CFG_FOCUSED_LAYERS == 'half':
            skipped_num_layers = int(num_layers * 0.5)
        elif CFG_FOCUSED_LAYERS == 'forth':
            skipped_num_layers = int(num_layers * 0.75)
        elif CFG_FOCUSED_LAYERS == 'last3':
            skipped_num_layers = int(num_layers - 3)
        else:
            raise NotImplementedError

        layer_tractions = dict()
        for layer_idx in range(skipped_num_layers, num_layers):
            lerp_ratio = (layer_idx + 1) / num_layers
            layer_argmax_base = torch.lerp(input_medium_base, output_argmax_base, lerp_ratio)
            layer_target_base = torch.lerp(input_medium_base, output_target_base, lerp_ratio)
            if self.cfg.ABL2_ESTIMATING_SEMANTIC == 'basis':
                layer_drift = (layer_target_base - 0).flatten()
            elif self.cfg.ABL2_ESTIMATING_SEMANTIC == 'delta':
                layer_drift = (layer_target_base - layer_argmax_base).flatten()
            else:
                raise NotImplementedError
            # unitize (to adjust to a normal range, which may vary by layer) it is necessary
            layer_drift_range = (torch.max(layer_drift) - torch.min(layer_drift))
            if layer_drift_range.abs() < 1e-8:
                continue
            scaled_layer_drift = layer_drift / layer_drift_range

            # 投影到旧update方向的正交空间
            projected = self._project_onto_history_orthogonal(layer_idx, scaled_layer_drift)
            layer_tractions[layer_idx] = projected

        return layer_tractions

    def plan_neurons(self, fn_editing, neurons, prompt, target_label: int, argmax_label: int):
        comparator = PriorityQueue(maxsize=0)
        for neuron in neurons:
            # simu editing (we merely measure the capability of neurons, so not require the 'factor')
            undo_fn = fn_editing(neurons=[neuron])
            # check the importance of neurons

            # target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(context + prefix, target)
            # matchness = Metric.same_accuracy(target_labels, argmax_labels)
            # # logger.debug(f'{matchness=}')

            step_logits = self._predict(prompt)
            _, _, step_probs, step_ranks = self._project(step_logits, [target_label, argmax_label])
            # undo neuron-editing
            undo_fn()
            step_target_prob, step_argmax_prob = step_probs

            # ...
            prob_gap = step_argmax_prob - step_target_prob
            objective_gap = prob_gap

            comparatee = (objective_gap, neuron)
            comparator.put(comparatee)

        # logger.warning('CHECK PQ')
        # pq = PriorityQueue(maxsize=0)
        # while not comparator.empty():
        #     comparatee = comparator.get()
        #     # logger.warning(f'{comparatee=}')
        #     pq.put(comparatee)
        return comparator

    def do_editing(self, prompt, target_label, argmax_label, medium_label=None, lr_scale=1):
        # load layer_tractions
        if medium_label is None:
            layer_tractions = self.compute_traction_simplified(target_label, argmax_label)
        else:
            layer_tractions = self.compute_traction(target_label, argmax_label, medium_label)

        # define functions
        # fn_editing = partial(lambda *args, **kwargs: lambda *a, **k: _)
        # ... when planning, we specify the factor = 0.01
        fn_editing = partial(self.modify_weights, layer_tractions=layer_tractions, factor=1e-2 * lr_scale)
        # ... do attribution once
        try:
            threshold, scores, neurons = self.attr.get_scores(prompt, target_label)
        except Exception as e:
            print(e)
            fair_neurons = list()
            return fair_neurons

        # we define a fn_stack for each token always
        fn_stack = deque()
        fair_neurons = list()
        while target_label != argmax_label:
            # not all cases could be easily corrected, we skip challenging one
            if len(fair_neurons) >= MAX_PATCHING_NUM:
                logger.warning(f'stopped since already edit too many neurons')
                while fn_stack:
                    undo_fn = fn_stack.pop()
                    undo_fn()
                # reset num_edit if it is an abandon case
                fair_neurons = list()
                break

            try:
                # make editing plans
                # ... we already find the most ideal neuron after each operation
                target_pq = self.plan_neurons(fn_editing, neurons, prompt, target_label, argmax_label)
                fair_comparatee = target_pq.get()
                objective_gap, neuron = fair_comparatee
                # ... only use the first neuron
                fair_neurons.append(neuron)
            except Exception as e:
                print(e)
                fair_neurons = list()
                break

            # conduct editing
            undo_fn = fn_editing(neurons=[neuron])
            fn_stack.append(undo_fn)
            # maybe rolling happened
            step_logits = self._predict(prompt)
            step_top_labels, step_top_probs, step_probs, step_ranks = self._project(
                step_logits, [target_label, argmax_label])
            # update argmax_label
            argmax_label = step_top_labels[0]

        # 将本次成功使用的 traction 记入历史
        if fair_neurons:
            self._register_history_tractions(layer_tractions)

        return fair_neurons

    def hack_model(self):
        # peft_config = LoraConfig(task_type="CAUSAL_LM")
        # self.model = get_peft_model(self.model, peft_config)
        for name, parameter in self.model.named_parameters():
            # parameter.requires_grad = self.attrs['layers'] in name
            parameter.requires_grad = self.attrs['ff_output'] + '.weight' in name
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for name, param in (self.model.named_parameters()):
            # logger.debug(f'{name=}, {param.requires_grad=}')
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        trainable_ratio = 100 * trainable_params / all_param
        logger.info(f"{trainable_params=} || {all_param=} || {trainable_ratio} %")

    def pipeline(self, source: str, target: str):
        # we reinit model to avoid interferes
        self._reinit()
        self.model.eval()

        # refresh
        target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(source, target)
        pre_gen_tokens = argmax_tokens

        post_gen_tokens = []
        oracle_tokens = []
        nums_skip = []
        nums_edit = []
        edited_neurons = []
        num_iter = len(target_labels)

        for idx, (target_label, argmax_label) in enumerate(zip(target_labels, argmax_labels)):
            logger.success('=' * 9 + f'{idx + 1}/{num_iter}' + '=' * 9)
            if 'codegen' in self.cfg.model_name:
                target_label = self.tokenizer.encode(' ')[0] if target_label > 50256 else target_label
                argmax_label = self.tokenizer.encode(' ')[0] if argmax_label > 50256 else argmax_label
            target_token = self.tokenizer.decode(target_label, skip_special_tokens=True)
            argmax_token = self.tokenizer.decode(argmax_label, skip_special_tokens=True)

            # we skip all correct predictions, and focus on these most needed positions
            if target_label == argmax_label:
                post_gen_tokens.append(argmax_token)
                oracle_tokens.append(target_token)
                continue

            prompt = source + ''.join(oracle_tokens)
            if self.cfg.ABL2_ESTIMATING_SEMANTIC == 'basis':
                input_encoding = self.tokenizer.encode_plus(prompt, add_special_tokens=False, truncation=True, return_tensors="pt")
                input_ids = input_encoding["input_ids"].to(DEVICE)
                medium_label = input_ids[0, -1].item()
            elif self.cfg.ABL2_ESTIMATING_SEMANTIC == 'delta':
                medium_label = None
            else:
                raise NotImplementedError

            # TODO when the last three parameters, our approach will be more effective, but slower...
            fair_neurons = self.do_editing(prompt, target_label, argmax_label, medium_label)
            num_edit = len(fair_neurons)
            num_skip = 1 if num_edit == 0 else 0

            if num_skip > 0:
                nums_skip.append(num_skip)
            if num_edit > 0:
                nums_edit.append(num_edit)
                edited_neurons.append(fair_neurons)

            # post-editing
            post_logits = self._predict(prompt)
            post_top_labels, post_top_probs, post_probs, post_ranks = self._project(
                post_logits, [target_label, argmax_label])
            if 'codegen' in self.cfg.model_name:
                post_top_labels = [self.tokenizer.encode(' ')[0] if label > 50256 else label for label in post_top_labels]
            post_top_tokens = [self.tokenizer.decode(label, skip_special_tokens=True) for label in post_top_labels]
            # update argmax_token
            argmax_token = post_top_tokens[0]
            post_gen_tokens.append(argmax_token)
            oracle_tokens.append(target_token)

        self.model.eval()
        logger.warning(f'{post_gen_tokens=}')
        target_tokens, argmax_tokens, target_labels, argmax_labels = self.generate(source, target)
        post_gen_tokens = argmax_tokens

        return pre_gen_tokens, post_gen_tokens, target_tokens, edited_neurons, nums_skip, nums_edit

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

    def process(self, prompt, target_label, argmax_label):
        # we skip all correct predictions, and focus on these most needed positions
        if target_label == argmax_label:
            if 'codegen' in self.cfg.model_name:
                target_label = self.tokenizer.encode(' ')[0] if target_label > 50256 else target_label
                argmax_label = self.tokenizer.encode(' ')[0] if argmax_label > 50256 else argmax_label
            target_token = self.tokenizer.decode(target_label, skip_special_tokens=True)
            argmax_token = self.tokenizer.decode(argmax_label, skip_special_tokens=True)
            logger.info(f'{target_token=} ({target_label}), {argmax_token=} ({argmax_label})')
            logger.warning(f'skip editing: the target_label is already ranked #0')
            # return 0, 0

        if self.cfg.ABL2_ESTIMATING_SEMANTIC == 'basis':
            input_encoding = self.tokenizer.encode_plus(prompt, add_special_tokens=False, truncation=True, return_tensors="pt")
            input_ids = input_encoding["input_ids"].to(DEVICE)
            medium_label = input_ids[0, -1].item()
        elif self.cfg.ABL2_ESTIMATING_SEMANTIC == 'delta':
            medium_label = None
        else:
            raise NotImplementedError

        # for benchmark study, we use a larger factor, (for example, 0.1, which is 10 times of 0.01)
        # this is to balance the effects of sgd_process to some extent...
        self.do_editing(prompt, target_label, argmax_label, medium_label, lr_scale=self.cfg.scale)
        # self.do_editing(prompt, target_label, argmax_label, medium_label)
        # fair_neurons = self.do_editing(prompt, target_label, argmax_label, medium_label)
        # num_edit = len(fair_neurons)
        # num_skip = 1 if num_edit == 0 else 0
        # return num_skip, num_edit

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
        range_g = len(nested_triplets)      # for index_g in range(range_g), index_g stays unchanged in iteration...
        range_s = len(nested_triplets[0])   # for index_s in range(range_s), index_s stays unchanged in iteration...

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

    @torch.no_grad()
    def modify_weights(
            self,
            layer_tractions: Dict[int, Tensor],
            neurons: List[Tuple[int]],
            factor=1.0,
    ) -> Callable:
        model_layers = get_attr(self.model, self.attrs['layers'])
        # backup original weights
        original_weights = []
        for (layer_idx, position) in neurons:
            ff_gen_weights = get_attr(model_layers[layer_idx], self.attrs['ff_output'] + '.weight')
            if "gpt2" in self.cfg.model_name:
                original_weights.append(ff_gen_weights[position, :].detach().clone())
            else:
                original_weights.append(ff_gen_weights[:, position].detach().clone())

        # modify the weights by subtracting the actual_embedding and adding the target_embedding
        for (layer_idx, position) in neurons:
            # logger.debug(f'{(layer_idx, position)=}')
            traction = layer_tractions[layer_idx] * factor
            ff_gen_weights = get_attr(model_layers[layer_idx], self.attrs['ff_output'] + '.weight')
            # logger.critical(f'{traction.size()=}')  # [1, 1024]
            # logger.critical(f'{ff_gen_weights[-1, :].size()=}')  # [4096]
            # logger.critical(f'{ff_gen_weights[:, -1].size()=}')  # [1024]
            # logger.critical(f'{ff_gen_weights.size()=}')  # [1024, 4096]
            if "gpt2" in self.cfg.model_name:
                ff_gen_weights[position, :] += traction
                ff_gen_weights[position, :] /= (1 + factor)
                # if pathways is not None:
                #     pathway = pathways[layer_idx]
                #     ff_gen_weights[position, pathway] += traction[pathway]
                #     ff_gen_weights[position, pathway] /= 2
                # else:
                #     ff_gen_weights[position, :] += traction
                #     ff_gen_weights[position, :] /= 2
            else:
                ff_gen_weights[:, position] += traction
                ff_gen_weights[:, position] /= (1 + factor)
                # if pathways is not None:
                #     pathway = pathways[layer_idx]
                #     ff_gen_weights[pathway, position] += traction[pathway]
                #     ff_gen_weights[pathway, position] /= 2
                # else:
                #     ff_gen_weights[:, position] += traction
                #     ff_gen_weights[:, position] /= 2

        # restore original weights
        @torch.no_grad()
        def undo_fn():
            for (layer_idx, position), original_weight in zip(neurons, original_weights):
                ff_gen_weights = get_attr(model_layers[layer_idx], self.attrs['ff_output'] + '.weight')
                if "gpt2" in self.cfg.model_name:
                    ff_gen_weights[position, :] = original_weight
                else:
                    ff_gen_weights[:, position] = original_weight

        return undo_fn
