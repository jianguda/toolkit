from functools import partial
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from captum._utils.typing import ModuleOrModuleList
from captum.attr import LayerGradientXActivation, LayerActivation
from loguru import logger
from torch import Tensor

from lm_lens.function.config.shared import DEVICE, CFG_FOCUSED_LAYERS, MAX_PATCHING_NUM
from lm_lens.function.utils import get_attr, obtain_loss


class Attr:
    def __init__(self, cfg, model, tokenizer, attrs):
        self.cfg = cfg
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        self.attrs = attrs

    def _get_embeddings(self, token_labels) -> torch.Tensor:
        # embeddings_matrix = attrgetter(self.attrs['embedding'])(self.model)
        embeddings_matrix = get_attr(self.model, self.attrs['embedding'])
        embeddings_matrix = embeddings_matrix.weight
        inputs_embeds = [embeddings_matrix[token_label] for token_label in token_labels]
        inputs_embeds = torch.stack(inputs_embeds)
        return inputs_embeds

    def get_layer_activations(
            self,
            forward_func: Callable,
            layer_attr: ModuleOrModuleList,
            inputs_tensor: torch.Tensor,
            target_label: int,
            side: str = 'output',
    ) -> Tensor:
            algo = LayerActivation(forward_func, layer_attr)
            attribution = algo.attribute(
                inputs_tensor,
                attribute_to_layer_input=(side == 'input'),
            )[0].detach()
            layer_actv = torch.norm(attribution, dim=0)
            # logger.debug(f'{attribution.size()=}')
            # logger.debug(f'{layer_actv.size()=}')
            return layer_actv

    def get_layer_scores(
            self,
            forward_func: Callable,
            layer_attr: ModuleOrModuleList,
            inputs_tensor: torch.Tensor,
            target_label: int,
            side: str = 'output',
    ) -> Tensor:
            algo = LayerGradientXActivation(forward_func, layer_attr)
            attribution = algo.attribute(
                inputs_tensor,
                target=target_label,
                attribute_to_layer_input=(side == 'input'),
            )[0].detach()
            layer_score = torch.norm(attribution, dim=0)
            # logger.debug(f'{attribution.size()=}')
            # logger.debug(f'{layer_score.size()=}')
            return layer_score

            # attributions = torch.stack(attributions, dim=0).sum(dim=0) / len(attributions)
            # print(f'{attributions.shape=}')
            # attributions = attributions.squeeze(0)
            # norm = torch.norm(attributions, dim=1)  # by num_inputs
            # norm = torch.norm(attributions, dim=-2)  # by num_neuron
            # normed_attributes = norm / torch.sum(norm)  # normalize the values to let their sum be 1
            # print(f'{normed_attributes=}')
            # print(f'{normed_attributes.shape=}')

    # TODO shall merge get_scores with get_layer_filters??? (there are two modes in get_layer_filters for RQ2...)
    # TODO also check hyperparameters...
    def get_scores(self, prompt: str, target_label: int, tabu_neurons=None):
        prompt_labels = [label for label in self.tokenizer.encode(prompt, add_special_tokens=False)]
        # logger.debug(f'{prompt_labels=}')
        prompt_inputs_embeds = self._get_embeddings(prompt_labels)
        prompt_inputs_tensor = prompt_inputs_embeds.unsqueeze(0)
        # logger.debug(f'{prompt_inputs_embeds=}')
        # logger.debug(f'{prompt_inputs_tensor=}')

        if 'codegen' in self.cfg.model_name:
            # hidden_size = self.model.config.n_embd
            intermediate_size = self.model.config.n_embd * 4
        else:
            # hidden_size = self.model.config.hidden_size
            intermediate_size = self.model.config.intermediate_size

        def model_forward(inputs_: torch.Tensor) -> torch.Tensor:
            output = self.model(inputs_embeds=inputs_)
            return F.softmax(output.logits[:, -1, :], dim=-1)
        forward_func = partial(model_forward)

        attribution_scores = list()
        model_layers = get_attr(self.model, self.attrs['layers'])
        num_layers = len(model_layers)
        # TODO which strategy of restriction to use???
        ff_input_attr = self.attrs['ff_input']
        ff_output_attr = self.attrs['ff_output']
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

        # located_pathways = defaultdict(list)
        for layer_idx in range(skipped_num_layers, num_layers):
            ffn_layer1_attr = get_attr(model_layers[layer_idx], ff_input_attr)  # 1024 * 4096
            ffn_layer2_attr = get_attr(model_layers[layer_idx], ff_output_attr)  # 4096 * 1024

            # neuron attribution
            # TODO layer2_input seems better, layer1_output is before the activation function (for example, it is GELU in CodeGEN)
            if self.cfg.ABL1_LOCATING_REFERENCE == 'actv':
                # For generalization, using activation is obviously better than using others (other attribution scores)
                layer_input_activations = self.get_layer_activations(
                    forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='input')
                attribution_scores.append(layer_input_activations)
            elif self.cfg.ABL1_LOCATING_REFERENCE == 'attr':
                layer_input_scores = self.get_layer_scores(forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='input')
                attribution_scores.append(layer_input_scores)
                # layer_output_scores = self.get_layer_scores(forward_func, ffn_layer1_attr, prompt_inputs_tensor, target_label, side='output')
                # attribution_scores.append(layer_output_scores)
            elif self.cfg.ABL1_LOCATING_REFERENCE == 'rand':
                layer_input_scores = torch.rand(intermediate_size)
                attribution_scores.append(layer_input_scores)
            else:
                raise NotImplementedError

            # # we use pathway attribution to limit the pathways to edit, to demonstrate generalization and specificity
            # # because more editing numbers is the precondition for more targeted editing
            # # reducing side effects (ordering of top cands and changes on their probs)
            # if self.cfg.ABL2_ESTIMATING_SCOPE == 'neuron':
            #     located_pathways = None
            # elif self.cfg.ABL2_ESTIMATING_SCOPE == 'pathway':
            #     # fine more contributed pathways
            #     most_contributed_pathways = list()
            #     # step * dim: (? * 4096) or (? * 1024) depends on it is 'input' side or 'output' side
            #     layer_output_scores = self.get_layer_scores(forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='output')
            #     layer_output_scores = layer_output_scores.flatten().cpu().float().numpy()
            #     for neuron_idx in np.argpartition(layer_output_scores, -EXTREME_PATHWAY_NUM)[-EXTREME_PATHWAY_NUM:]:
            #         # pathway = (layer_idx, neuron_idx)
            #         # most_contributed_pathways.append(pathway)
            #         most_contributed_pathways.append(neuron_idx)
            #     located_pathways[layer_idx] = most_contributed_pathways
            # else:
            #     raise NotImplementedError

        # pad the attribution score matrix
        attribution_scores = torch.stack(attribution_scores).to(DEVICE)
        # attribution_scores = torch.abs(attribution_scores).to(DEVICE)
        pad_attribution_scores = torch.zeros(skipped_num_layers, len(attribution_scores[0])).to(DEVICE)
        attribution_scores = torch.cat([pad_attribution_scores, attribution_scores], dim=0)
        # logger.critical(f'{attribution_scores.size()=}')  # [20, 4096]

        # tabu_neurons (it works!!!)
        if tabu_neurons is not None:
            for tabu_neuron in tabu_neurons:
                # logger.debug(f'{tabu_neuron=}')
                # logger.debug(f'{attribution_scores[tabu_neuron]=}')
                attribution_scores[tabu_neuron] = 0.0
                # logger.debug(f'{attribution_scores[tabu_neuron]=}')

        # the neuron amount is 4096 neurons/layer * 20 layers
        # find globally most contributed neurons to outputs to filter neurons
        # globally performs better than layer-ly!!!
        # TODO actually, we will further find most critical neurons from the pool
        filter_ratio_quantile = (1 - MAX_PATCHING_NUM / (intermediate_size * num_layers))
        scores = attribution_scores.flatten().cpu().float().numpy()
        # in rare cases (caused by attribution), the scores can be zeros, then we shall skip the editing
        threshold = np.quantile(scores, filter_ratio_quantile)
        most_contributed_neurons = torch.nonzero(attribution_scores >= threshold).cpu().tolist()
        most_contributed_neurons = [tuple(neuron) for neuron in most_contributed_neurons]
        located_neurons = most_contributed_neurons
        return threshold, attribution_scores, located_neurons

    # attribution (target_label => argmax_label?)
    def get_layer_filters(self, prompt_labels, target_label: int):
        # prompt_labels = [label for unit in prompt.split() for label in self.tokenizer.encode(unit)]
        # logger.debug(f'{prompt_labels=}')
        prompt_inputs_tensor = self._get_embeddings(prompt_labels)
        # logger.debug(f'{prompt_inputs_embeds=}')
        # logger.debug(f'{prompt_inputs_tensor.shape=}')  # =torch.Size([1, 11, 512])
        # TODO if we use attribution on the -batch mode (seems meaningless), inputs shall be a list of tensors...

        def model_forward(inputs_: torch.Tensor) -> torch.Tensor:
            output = self.model(inputs_embeds=inputs_)
            return F.softmax(output.logits[:, -1, :], dim=-1)
        forward_func = partial(model_forward)

        model_layers = get_attr(self.model, self.attrs['layers'])
        num_layers = len(model_layers)
        ff_output_attr = self.attrs['ff_output']

        if self.cfg.ABL2_ABLATION_WHICH == 'autoly':
            ratio_quantile = (1 - self.cfg.ABL2_ABLATION_RATIO / num_layers)
            # the neuron amount is 4096 neurons/layer * 20 layers
            # find globally most contributed neurons to outputs to filter neurons
            # ...
            attribution_scores = list()
            for layer_idx in range(num_layers):
                ffn_layer2_attr = get_attr(model_layers[layer_idx], ff_output_attr)  # 4096 * 1024
                # we directly use the attribution method
                layer_attribution_scores = self.get_layer_scores(forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='input')
                attribution_scores.append(layer_attribution_scores)
                # layer_output_scores = self.get_layer_scores(forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='output')
                # attribution_scores.append(layer_output_scores)

            # globally: let ratio_quantile work on the model ...
            attribution_scores = torch.stack(attribution_scores)
            # attribution_scores = torch.abs(attribution_scores)
            # logger.warning(f'{attribution_scores.size()=}')  # [20, 4096]
            threshold = np.quantile(attribution_scores.cpu().float().numpy(), ratio_quantile)
            layer_filters = torch.where(attribution_scores >= threshold, 1.0, 0.0)
            nonzero_neurons = torch.nonzero(layer_filters >= threshold).cpu().tolist()
            located_neurons = [tuple(neuron) for neuron in nonzero_neurons]
            # logger.warning(f'{len(nonzero_neurons)=}')
            # logger.debug(f'{layer_filters.shape=}')
        # elif self.cfg.ABL2_ABLATION_WHICH == 'evenly':
        #     ratio_quantile = (1 - self.cfg.ABL2_ABLATION_RATIO / num_layers)
        #     # even-ly: let ratio_quantile work on each layer ...
        #     located_neurons = list()
        #     # layer_parameter_filters = list()
        #     for layer_idx in range(num_layers):
        #         # ...
        #         ffn_layer2_attr = get_attr(model_layers[layer_idx], ff_output_attr)  # 4096 * 1024
        #         # we directly use the attribution method
        #         layer_attribution_scores = self.get_layer_scores(forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='input')
        #
        #         # logger.warning(f'{layer_attribution_scores.size()=}')  # [1, 4096]
        #         threshold = np.quantile(layer_attribution_scores.cpu().float().numpy(), ratio_quantile)
        #         layer_parameter_filter = torch.where(layer_attribution_scores >= threshold, 1.0, 0.0)
        #
        #         nonzero_neurons = torch.nonzero(layer_parameter_filter >= threshold).cpu().tolist()
        #         layer_located_neurons = [tuple((layer_idx, neuron[0])) for neuron in nonzero_neurons]
        #         located_neurons.extend(layer_located_neurons)
        #         # logger.warning(f'{len(nonzero_neurons)=}')
        #         # layer_parameter_filters.append(layer_parameter_filter)
        #     # layer_filters = torch.stack(layer_parameter_filters)
        elif self.cfg.ABL2_ABLATION_WHICH == 'layerly':
            ratio_quantile = (1 - self.cfg.ABL2_ABLATION_RATIO)
            # layer-ly: first attribute the layer and then use ratio_quantile to filter the neurons...
            input_ids = prompt_labels
            # logger.debug(f'{input_ids.shape=}')
            labels = torch.tensor(target_label).to(DEVICE)
            # logger.debug(f'{labels.shape=}')

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
            layer_idx = layer_losses.index(max(layer_losses))
            # ...
            ffn_layer2_attr = get_attr(model_layers[layer_idx], ff_output_attr)  # 4096 * 1024
            # we directly use the attribution method
            layer_attribution_scores = self.get_layer_scores(forward_func, ffn_layer2_attr, prompt_inputs_tensor, target_label, side='input')
            # layer_attribution_scores = torch.abs(layer_attribution_scores)

            # logger.warning(f'{layer_attribution_scores.size()=}')  # [1, 4096]
            threshold = np.quantile(layer_attribution_scores.cpu().float().numpy(), ratio_quantile)
            layer_parameter_filter = torch.where(layer_attribution_scores >= threshold, 1.0, 0.0)

            nonzero_neurons = torch.nonzero(layer_parameter_filter >= threshold).cpu().tolist()
            located_neurons = [tuple((layer_idx, neuron[0])) for neuron in nonzero_neurons]
        else:
            raise NotImplementedError

        # # logger.debug(f'{located_neurons=}')
        # logger.debug(f'{len(located_neurons)=}')

        # logger.warning(f'{ratio_quantile=}')
        return located_neurons
