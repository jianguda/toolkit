# Model Repair Module

This module integrates the STAR (Semantic Targeting for Analytical Repair) functionality into LM Lens, enabling visualization of model repair processes.

## Supported Repair Approaches

The module supports four repair approaches:

1. **mint**: MINT baseline method - uses neuron editing with orthogonal projection
2. **me-sgd**: SGD-based iterative repair - uses standard SGD optimization
3. **me-iter**: Iterative repair - uses semantic steering with iterative updates
4. **me-batch**: Batch repair - processes multiple tokens in batch mode

## Usage

### In the UI

1. Enable repair mode by checking "Enable repair" in the sidebar
2. Select a repair approach from the dropdown
3. Set the number of training epochs
4. Enter the source prompt and target output
5. Click "Run Repair" to start the repair process

### Programmatic Usage

```python
from lm_lens.function.repair_interface import RepairInterface

# Initialize repair interface
repair = RepairInterface(
    model_name="Salesforce/codegen-350M-multi",
    scale=1,
    data_name="demo"
)

# Run repair
result = repair.repair(
    source="def hello():",
    target="def hello():\n    print('Hello, world!')",
    approach="me-iter",
    epoch_num=10
)

# Access results
print(f"Before: {result.pre_gen_tokens}")
print(f"After: {result.post_gen_tokens}")
print(f"Target: {result.target_tokens}")
```

## Repair Result Structure

The `RepairResult` dataclass contains:

- `pre_gen_tokens`: Tokens generated before repair
- `post_gen_tokens`: Tokens generated after repair
- `target_tokens`: Target tokens
- `edited_neurons`: List of edited neurons (for MINT approach)
- `nums_skip`: Number of skipped tokens (for MINT approach)
- `nums_edit`: Number of edited neurons per token (for MINT approach)

## Integration with Visualization

The repair results are automatically displayed in the main UI, showing:
- Before/After/Target comparison
- Token-level accuracy metrics
- Edited neurons visualization (for MINT approach)
- Neuron editing statistics

