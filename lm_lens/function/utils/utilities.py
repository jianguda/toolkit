import time

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import transformers
from loguru import logger

from lm_lens.function.config.shared import CFG_PROMPT_STYLE, DEVICE


def get_attr(mod: nn.Module, attrs: str):
    for attr in attrs.split("."):
        mod = getattr(mod, attr)
    return mod


def set_attr(mod: nn.Module, attrs: str, new_mod: nn.Module):
    for attr in attrs.split(".")[:-1]:
        mod = getattr(mod, attr)
    setattr(mod, attrs.split(".")[-1], new_mod)


def freeze_params(model, attrs, layers=None):
    param_groups = []
    for name, param in model.named_parameters():
        # logger.debug(f'{name=}')
        if attrs['ff_output'] in name:
            if 'weight' in name:
                if layers is None:
                    param.requires_grad = True
                else:
                    xpath_name = name.removeprefix(attrs['layers'] + '.')
                    xpath_address = xpath_name.split('.')
                    layer_index = int(xpath_address[0])
                    param.requires_grad = layer_index in layers
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
        param_groups.append({'params': [param]})
    return param_groups


# based on our experiments, max-min scaling is better than mean-std scaling
def compute_sss(lm_head_matrix, target_label, argmax_label):
    output_target_embed = lm_head_matrix[target_label].detach()
    output_argmax_embed = lm_head_matrix[argmax_label].detach()
    semantic_steer = output_target_embed - output_argmax_embed
    # # ...
    # median = torch.median(semantic_steer)
    # iqr = torch.quantile(semantic_steer, 0.75) - torch.quantile(semantic_steer, 0.25)
    # semantic_steer = (semantic_steer - median) / (iqr + 1e-8)
    # ...
    # median = torch.median(semantic_steer)
    iqr = torch.quantile(semantic_steer, 0.75) - torch.quantile(semantic_steer, 0.25)
    semantic_steer =  semantic_steer / (iqr + 1e-8)

    if len(semantic_steer.shape) == 1:
        semantic_steer = semantic_steer.unsqueeze(0)
    # print(f'{scaled_semantic_steer.shape=}')
    # scaled_semantic_steer.shape=torch.Size([1024])
    # TODO we plan to no longer scale the ss, so the LR may be different...
    return semantic_steer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        # logger.debug(f'{name=}, {param.requires_grad=}')
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_ratio = 100 * trainable_params / all_param
    logger.info(f"{trainable_params=} || {all_param=} || {trainable_ratio} %")


# ...
def obtain_loss(model, input_ids, labels, is_batch=True):
    if is_batch:
        labels = labels.unsqueeze(0)
        combined_ids = torch.cat([input_ids, labels], dim=-1)
        ground_truth = combined_ids.clone()
        # Set the labels for the input part to -100 (ignored in loss computation)
        ground_truth[:, :input_ids.size(1)] = -100
        # logger.debug(f'{ground_truth=}')
        outputs = model(input_ids=combined_ids, labels=ground_truth, output_hidden_states=True, return_dict=True)
        loss = outputs.loss
    else:
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

        # cross-entropy loss
        logits = outputs.logits[:, -1, :]
        # logger.debug(f'{logits.shape=}')
        # logger.debug(f'{truth.shape=}')
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        # simi-based ...
        # hidden_state = outputs.hidden_states[-1]
        # embeds_hat = hidden_state[:, -1, :]
        # simis = F.cosine_similarity(embeds_hat, self.lm_head_matrix, dim=-1)
        # logits = simis.to(DEVICE)
        # target_label = ground_truth.item()
        # anchor = self.lm_head_matrix[target_label]
        # loss = 1 - F.cosine_similarity(embeds_hat, anchor, dim=-1)  #.detach()
        # loss = (torch.max(simis) - simis[target_label])  #.detach()
    return loss


# TODO currently it is not that useful...
# Example usage:
# Z = torch.randn(1, 400)
# S = torch.randn(1, 100)
# zero_rows = [0, 10, 20, 30]  # Rows to be set to zero
# W = constrained_lstsq(Z, S, zero_rows)
def constrained_lstsq(Z, S, nonzero_rows: list=None):
    # Z * W = S
    # for example: Z shape = (1, 400), W shape = (400, 100), S shape = (1, 100)
    # nonzero_rows: the row indices of W to be set to nonzero

    # Step 1: Solve unconstrained problem
    # W = torch.linalg.lstsq(Z, S).solution
    # Step 2: Set specified rows to zero
    # W[zero_rows, :] = 0
    W = torch.zeros((Z.shape[-1], S.shape[-1]))
    # Step 3: Solve for remaining rows
    Z_reduced = Z[:, nonzero_rows]
    W_reduced = torch.linalg.lstsq(Z_reduced, S).solution
    W = W.to(DEVICE)
    W_reduced = W_reduced.to(DEVICE)
    # Update W with the new solution for non-zero rows
    W[nonzero_rows, :] = W_reduced
    return W


###
def stabilize(reproducibility=True, seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def print_info(model):
    print(model)
    for name, parameter in model.base_model.named_parameters():
        print(name, parameter.size())


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        results = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = format_score(end - start)
        logger.debug(f"{func.__name__} takes {elapsed_time} seconds")
        return elapsed_time, results

    return wrapper


def format_score(datum):
    return round(datum, 3)


def format_rate(datum):
    return f'{format_score(datum * 100)}%'


def format_ratio(pre_datum, post_datum):
    sign_prefix = ('+' if post_datum >= pre_datum else '')
    ratio = sign_prefix + f'{format_score((post_datum - pre_datum) * 100)}%'
    # abs_ratio = sign_prefix + f'{format_score((post_datum - pre_datum) * 100)}%'
    # rel_ratio = sign_prefix + f'{format_score((post_datum / pre_datum - 1.) * 100)}%'
    return ratio
