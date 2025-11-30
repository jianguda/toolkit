from collections import defaultdict
from difflib import SequenceMatcher
from functools import partial
from statistics import fmean

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from loguru import logger
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    Starcoder2ForCausalLM, CodeGenForCausalLM
)

from lm_lens.function.config.shared import (
    DRYRUN_TEST_NUM, VOCAB_OUTS_DIR, SIMI_OUTS_DIR, MATRIX_TEMP, VOCAB_TEMP, DATA_DIR,
    SHOT_DEMO_NUM, CFG_MODE_DRYRUN, GEN_TEMP, GEN_OUTS_DIR
)
from lm_lens.function.utils import format_score, format_ratio, format_rate


class NeoLoader:
    @staticmethod
    def load_corpus(data_name):
        # in the future, we may turn to use EvalPlus (HumanEval+, MBPP+, BigcodeBench+, ...)
        if data_name == 'demo':
            text_data = ['The countries of the European Union are:\n1. Austria\n2. Belgium\n3. Bulgaria\n4.']
            code_data = ['Denmark']
            # allow_vocab_labels = ['Denmark']
        elif data_name == 'bigcode':
            text_data, code_data = _load_bcbh()
        elif data_name == 'human':
            text_data, code_data = _load_he()
            # allow_vocab_labels = NeoLoader.load_allow_vocab(data_name, chunk_id, model_name, solutions)
        else:
            raise NotImplementedError
        if CFG_MODE_DRYRUN:
            text_data = text_data[:DRYRUN_TEST_NUM]
            code_data = code_data[:DRYRUN_TEST_NUM]

        input_lens = [len(text) for text in text_data]
        output_lens = [len(code) for code in code_data]
        avg_input_lens = fmean(input_lens)
        avg_output_lens = fmean(output_lens)
        print(f'{avg_input_lens=}, {avg_output_lens=}')
        return text_data, code_data

    @staticmethod
    def retrieve_top_neighbors(simi_matrix, test_texts, num_neighbors=SHOT_DEMO_NUM):
        simi_neighbors = dict()
        for idx, test_text in enumerate(test_texts):
            similarities = simi_matrix[idx]
            neighbor_ids = np.argpartition(similarities, -num_neighbors)[-num_neighbors:]
            simi_neighbors[idx] = neighbor_ids
        return simi_neighbors

    @staticmethod
    def load_tokenizer(model_name: str):
        if "gpt2" in model_name:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "codegen" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "starcoder" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif "CodeLlama" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif "OpenCoder" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        elif "CodeQwen" in model_name or "Qwen2.5-Coder" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Model {model_name} not supported")
        return tokenizer

    @staticmethod
    def load_model(model_name: str):
        config = AutoConfig.from_pretrained(model_name, output_scores=True)
        # config = AutoConfig.from_pretrained(
        #     model_name,
        #     device_map='auto',
        #     output_scores=True,
        #     output_hidden_states=True,
        #     # trust_remote_code=True,
        #     load_in_4bit=(DEVICE == 'cuda')
        # )
        attrs = dict()
        if "gpt2" in model_name:
            attrs['layers'] = 'transformer.h'
            attrs['ff_act'] = 'mlp.act'
            attrs['ff_input'] = 'mlp.c_fc'
            attrs['ff_output'] = 'mlp.c_proj'
            attrs['embedding'] = 'transformer.wte'
            attrs['lm_head'] = 'lm_head'
            model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        elif "codegen" in model_name:
            attrs['layers'] = 'transformer.h'
            attrs['ff_act'] = 'mlp.act'
            attrs['ff_input'] = 'mlp.fc_in'
            attrs['ff_output'] = 'mlp.fc_out'
            attrs['embedding'] = 'transformer.wte'
            attrs['lm_head'] = 'lm_head'
            model = CodeGenForCausalLM.from_pretrained(model_name, config=config)
        elif "starcoder" in model_name:
            attrs['layers'] = 'model.layers'
            attrs['ff_act'] = 'mlp.act'
            attrs['ff_input'] = 'mlp.c_fc'
            attrs['ff_output'] = 'mlp.c_proj'
            attrs['embedding'] = 'model.embed_tokens'
            attrs['lm_head'] = 'lm_head'
            model = Starcoder2ForCausalLM.from_pretrained(model_name, config=config)
        elif "CodeLlama" in model_name:
            attrs['layers'] = 'model.layers'
            attrs['ff_act'] = 'mlp.act_fn'
            attrs['ff_input'] = 'mlp.up_proj'
            attrs['ff_output'] = 'mlp.down_proj'
            attrs['embedding'] = 'model.embed_tokens'
            attrs['lm_head'] = 'lm_head'
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        elif "OpenCoder" in model_name:
            attrs['layers'] = 'model.layers'
            attrs['ff_act'] = 'mlp.act_fn'
            attrs['ff_input'] = 'mlp.up_proj'
            attrs['ff_output'] = 'mlp.down_proj'
            attrs['embedding'] = 'model.embed_tokens'
            attrs['lm_head'] = 'lm_head'
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
        elif "Qwen" in model_name and "Coder" in model_name:
            attrs['layers'] = 'model.layers'
            attrs['ff_act'] = 'mlp.act_fn'
            attrs['ff_input'] = 'mlp.up_proj'
            attrs['ff_output'] = 'mlp.down_proj'
            attrs['embedding'] = 'model.embed_tokens'
            attrs['lm_head'] = 'lm_head'
            # we study scaling with Qwen-Coder models, which are loaded in BF16 to save memory
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, config=config)
            # if '32B' in model_name:
            #     model = FSDP(model, sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD)
            #     model.gradient_checkpointing_enable()
        else:
            raise ValueError(f"Model {model_name} not supported")
        model.eval()
        # disable gradients
        # for param in model.parameters():
        #     param.requires_grad = False
        return config, model, attrs


def _load_bcbh():
    dataset = load_dataset('bigcode/bigcodebench-hard', split='v0.1.4', streaming=True)

    prompts = []
    solutions = []
    for datum in dataset:
        # prompt = datum["complete_prompt"]  # for base LMs
        # prompt = datum["instruct_prompt"]  # for chat LMs
        prompt = datum["code_prompt"]
        solution = datum["canonical_solution"]
        solution = solution.replace(' ' * 4, '\t')
        prompts.append(prompt)
        solutions.append(solution)
    return prompts, solutions


def _load_he():
    dataset = load_dataset('openai_humaneval', split='test')

    prompts = []
    solutions = []
    # entry_points = []
    # test_funcs = []
    for datum in dataset:
        prompt = datum["prompt"]
        solution = datum["canonical_solution"]
        solution = solution.replace(' ' * 4, '\t')
        # entry_point = datum["entry_point"]
        # test_func = datum["test"]
        prompts.append(prompt)
        solutions.append(solution)
        # entry_points.append(entry_point)
        # test_funcs.append(test_func)
    return prompts, solutions


def well_load(func, folder, filename):
    import pickle
    if (folder / filename).is_file():
        with open(folder / filename, 'rb') as handle:
            data = pickle.load(handle)
    else:
        data = func()
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / filename, 'wb') as handle:
            pickle.dump(data, handle)
    return data


class Metric:
    @staticmethod
    def longest_match(gens: list[list[str]], refs: list[list[str]]):
        """Longest Common Substring on the token-level"""
        scores = list()
        for gen, ref in zip(gens, refs):
            gen, ref = tuple(gen), tuple(ref)
            matcher = SequenceMatcher(a=gen, b=ref, autojunk=False)
            match = matcher.find_longest_match()
            score = match.size / len(ref)
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def exact_matching(gens: list[list[str]], refs: list[list[str]]):
        """Exact Match on the token-level"""
        scores = list()
        metric = load('exact_match')
        # print(metric.inputs_description)
        for gen, ref in zip(gens, refs):
            score = metric.compute(predictions=gen, references=ref)['exact_match']
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def bleu_score(predictions: list[str], references: list[str]):
        """BLEU on the char-level"""
        predictions = [prediction for prediction in predictions]
        references = [[reference] for reference in references]
        metric = load('bleu')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['bleu']
        return format_score(score)

    @staticmethod
    def meteor_score(predictions: list[str], references: list[str]):
        """METEOR-1.0 on the char-level"""
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        metric = load('meteor')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['meteor']
        return format_score(score)

    @staticmethod
    def rouge_score(predictions: list[str], references: list[str]):
        """ROUGE on the char-level"""
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        metric = load('rouge')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['rougeL']
        return format_score(score)

    @staticmethod
    def edit_similarity(gens: list[str], refs: list[str]):
        """Edit Similarity on the char-level"""
        from nltk import edit_distance
        scores = list()
        for gen, ref in zip(gens, refs):
            score = 1 - edit_distance(gen, ref) / max(len(gen), len(ref))
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def exact_match_score(predictions: list[str], references: list[str]):
        """Exact Match on the char-level"""
        predictions = [prediction for prediction in predictions]
        references = [reference for reference in references]
        metric = load('exact_match')
        # print(metric.inputs_description)
        score = metric.compute(predictions=predictions, references=references)['exact_match']
        return format_score(score)

    @staticmethod
    def gestalt_match_score(gens: list[str], refs: list[str]):
        """Gestalt Pattern Matching on the token-level"""
        # https://github.com/python/cpython/blob/main/Lib/difflib.py#L597
        scores = list()
        for gen, ref in zip(gens, refs):
            gen, ref = tuple(gen), tuple(ref)
            matcher = SequenceMatcher(a=gen, b=ref, autojunk=False)
            score = matcher.ratio()
            scores.append(score)
        avg_score = np.average(scores)
        return format_score(avg_score)

    @staticmethod
    def contrast_scoring(pre_gens, post_gens, oracle_gens):
        # from collections import Counter
        # pre_cnt = Counter()
        # post_cnt = Counter()
        # for pre_seq, post_seq, oracle_seq in zip(pre_gens, post_gens, oracle_gens):
        #     for pre_token, post_token, oracle_token in zip(pre_seq, post_seq, oracle_seq):
        #         pre_cnt[pre_token == oracle_token] += 1
        #         post_cnt[post_token == oracle_token] += 1
        # logger.info(f'{pre_cnt=}')
        # logger.info(f'{post_cnt=}')
        # logger.info(f'{post_cnt[True] - pre_cnt[True]=}')
        # logger.info(f'{post_cnt[True] + post_cnt[False]=}')
        # ...
        for func in (
            Metric.exact_matching,
            # Metric.gestalt_matching,
        ):
            pre_score = func(pre_gens, oracle_gens)
            post_score = func(post_gens, oracle_gens)
            ratio = format_ratio(pre_score, post_score)
            logger.success(f'{func.__name__}: {pre_score=}, {post_score=} ({ratio=})')

    @staticmethod
    def contrast_scoring2(pre_gens, post_gens, oracle_gens):
        pre_gens = [' '.join(gen) for gen in pre_gens]
        post_gens = [' '.join(gen) for gen in post_gens]
        oracle_gens = [' '.join(gen) for gen in oracle_gens]
        for func in (
            Metric.edit_similarity,
            # Metric.exact_match_score,
            # Metric.gestalt_match_score,
            Metric.bleu_score,
            Metric.rouge_score,  # ROUGE is less recommended than BLEU
            # Metric.meteor_score,
        ):
            pre_score = func(pre_gens, oracle_gens)
            post_score = func(post_gens, oracle_gens)
            ratio = format_ratio(pre_score, post_score)
            logger.success(f'{func.__name__}: {pre_score=}, {post_score=} ({ratio=})')
