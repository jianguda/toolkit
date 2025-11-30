from pathlib import Path

import torch

# features
CFG_MODE_DRYRUN = False
CFG_MODE_BENCHMARK = False

dryrun_mark = '_.' if CFG_MODE_DRYRUN else ''
special_mark = '#benchmark.' if CFG_MODE_BENCHMARK else ''

DATA_DIR = Path.cwd().parent / 'data'
OUTS_DIR = Path.cwd().parent / 'outs'
SIMI_OUTS_DIR = OUTS_DIR / 'simi'
VOCAB_OUTS_DIR = OUTS_DIR / 'vocab'
LOG_OUTS_DIR = OUTS_DIR / (special_mark + 'log')
GEN_OUTS_DIR = OUTS_DIR / (special_mark + 'gen')
FIG_OUTS_DIR = OUTS_DIR / (special_mark + 'fig')
STATS_OUTS_DIR = OUTS_DIR / (special_mark + 'stats')
REPORT_OUTS_DIR = OUTS_DIR / (special_mark + 'report')

MATRIX_TEMP = dryrun_mark + 'matrix.{data_name}.{train_num}.{test_num}.pkl'
VOCAB_TEMP = dryrun_mark + 'vocab.{data_name}.{model_name}.pkl'
LOG_TEMP = dryrun_mark + 'me.{data_name}.{model_name}.log'
GEN_TEMP = dryrun_mark + 'me.{data_name}.{gen_mark}.jsonl'

# ========== EXPERIMENT CONFIGURATION ==========
# * choose one ('colon' seems quicker)
CFG_PROMPT_STYLE = 'colon'  # 'colon': separated by ':'
# CFG_PROMPT_STYLE = 'lines'  # 'lines': separated by '\n'

# * choose one ('full' takes longer time, 'half' seems most ideal, 'forth' and 'last3' could affect results)
# CFG_FOCUSED_LAYERS = 'all'  # 'all': x 42.993 seconds 30.427 seconds
CFG_FOCUSED_LAYERS = 'half'  # 'half': x 36.274 seconds 20.593 seconds
# CFG_FOCUSED_LAYERS = 'forth'  # 'forth': x 31.701 seconds 19.205 seconds
# CFG_FOCUSED_LAYERS = 'last3'  # 'last3': x 28.99 seconds 12.606 seconds

# dryrun settings
DRYRUN_TEST_NUM = 5
SHOT_DEMO_NUM = 1

# we mainly care about details of the top10 tokens
MAX_FOCUSING_NUM = 10
# patch at most 10 times in solving each failure
MAX_PATCHING_NUM = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL_REGISTRY = {
    'gpt2': 'gpt2',
    'gpt2-xl': 'gpt2-xl',
    # codegen
    'codegen': 'Salesforce/codegen-350M-multi',          # 20 layers * (1024 * 4) neurons/layer
    'codegen-2b': 'Salesforce/codegen-2B-multi',         # 32 layers * (2560 * 4) neurons/layer *
    # codegen2 (low-quality models)
    'codegen2-1b': 'Salesforce/codegen2-1B',          # 16 layers * (2048 * 4) neurons/layer
    'codegen2-3.7b': 'Salesforce/codegen2-3_7B',      # 16 layers * (4096 * 4) neurons/layer
    'codegen2-7b': 'Salesforce/codegen2-7B',          # 32 layers * (4096 * 4) neurons/layer
    'codegen25-7b': 'Salesforce/codegen25-7b-multi',  # 32 layers * (4096 * 4) neurons/layer
    # code llama * 32016 32016 32000 32016 (BF16)
    'codellama-7b': 'meta-llama/CodeLlama-7b-hf',       # 32 layers * (4096 * 4) neurons/layer *
    'codellama-13b': 'meta-llama/CodeLlama-13b-hf',     # 40 layers * (5120 * 4) neurons/layer !!!
    'codellama-34b': 'meta-llama/CodeLlama-34b-hf',     # 48 layers * (8192 * 4) neurons/layer !!!
    'codellama-70b': 'meta-llama/CodeLlama-70b-hf',     # 80 layers * (8192 * 4) neurons/layer !!!
    # starcoder2 * 49152 (F32, BF16, F32)
    'starcoder2-3b': 'bigcode/starcoder2-3b',           # 30 layers * (3072 * 4) neurons/layer
    'starcoder2-7b': 'bigcode/starcoder2-7b',           # 32 layers * (4608 * 4) neurons/layer
    'starcoder2-15b': 'bigcode/starcoder2-15b',         # 40 layers * (6144 * 4) neurons/layer !!!
    # opencoder * (F32 + F16)
    'opencoder-1.5b': 'infly/OpenCoder-1.5B-Base',
    'opencoder-8b': 'infly/OpenCoder-8B-Base',
    # qwen2.5-coder 151936 * 3 152064 * 3 (BF16)
    # (qwen2.5-coder is at the same period with opencoder ...)
    'qwencoder-0.5b': 'Qwen/Qwen2.5-Coder-0.5B',        # 24 layers * (896 * 4) neurons/layer !!!
    'qwencoder-1.5b': 'Qwen/Qwen2.5-Coder-1.5B',        # 28 layers * (1536 * 4) neurons/layer !!!
    'qwencoder-3b': 'Qwen/Qwen2.5-Coder-3B',            # 36 layers * (2048 * 4) neurons/layer !!!
    'qwencoder-7b': 'Qwen/Qwen2.5-Coder-7B',            # 28 layers * (3584 * 4) neurons/layer !!!
    'qwencoder-14b': 'Qwen/Qwen2.5-Coder-14B',          # 48 layers * (5120 * 4) neurons/layer !!!
    'qwencoder-32b': 'Qwen/Qwen2.5-Coder-32B',          # 64 layers * (5120 * 4) neurons/layer !!!
}
