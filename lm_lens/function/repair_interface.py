"""Interface for LM repair functionality with visualization support."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch

from lm_lens.function.config import Configure
from lm_lens.function.core.star_core import StarCore
from lm_lens.function.core.mint_core import MintCore


@dataclass
class RepairResult:
    """Result of a repair operation."""
    pre_gen_tokens: List[str]
    post_gen_tokens: List[str]
    target_tokens: List[str]
    edited_neurons: Optional[List[List[Tuple[int, int]]]] = None
    nums_skip: Optional[List[int]] = None
    nums_edit: Optional[List[int]] = None
    repair_steps: Optional[List[Dict[str, Any]]] = None


class RepairInterface:
    """Interface for model repair with different approaches."""
    
    def __init__(self, model_name: str, scale: int = 1, data_name: str = 'demo'):
        """Initialize repair interface.
        
        Args:
            model_name: Name of the model to repair
            scale: Scale factor for learning rate
            data_name: Name of the dataset
        """
        self.model_name = model_name
        self.scale = scale
        self.data_name = data_name
        self._core_instances = {}  # Cache cores by approach
        self._mint = None
    
    def _get_core(self, approach: str) -> StarCore:
        """Get or create StarCore instance for the given approach."""
        if approach not in self._core_instances:
            subtask_code = f'approach.0.{approach}'
            cfg = Configure(self.scale, self.data_name, self.model_name, subtask_code)
            self._core_instances[approach] = StarCore(cfg)
        return self._core_instances[approach]
    
    def _get_mint(self) -> MintCore:
        """Get or create MintCore instance."""
        if self._mint is None:
            subtask_code = 'approach.0.mint'
            cfg = Configure(self.scale, self.data_name, self.model_name, subtask_code)
            self._mint = MintCore(cfg)
        return self._mint
    
    def repair(
        self,
        source: str,
        target: str,
        approach: str = 'me-iter',
        epoch_num: int = 10
    ) -> RepairResult:
        """Repair model to generate target from source.
        
        Args:
            source: Source prompt
            target: Target output
            approach: Repair approach ('mint', 'me-sgd', 'me-iter', 'me-batch')
            epoch_num: Number of training epochs
            
        Returns:
            RepairResult with repair information
        """
        if approach == 'mint':
            return self._repair_mint(source, target)
        elif approach in ['me-sgd', 'me-iter', 'me-batch']:
            return self._repair_core(source, target, approach, epoch_num)
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def _repair_mint(self, source: str, target: str) -> RepairResult:
        """Repair using MINT approach."""
        mint = self._get_mint()
        pre_gen_tokens, post_gen_tokens, target_tokens, edited_neurons, nums_skip, nums_edit = mint.pipeline(source, target)
        
        return RepairResult(
            pre_gen_tokens=pre_gen_tokens,
            post_gen_tokens=post_gen_tokens,
            target_tokens=target_tokens,
            edited_neurons=edited_neurons,
            nums_skip=nums_skip,
            nums_edit=nums_edit
        )
    
    def _repair_core(self, source: str, target: str, approach: str, epoch_num: int) -> RepairResult:
        """Repair using Core approaches (me-sgd, me-iter, me-batch)."""
        core = self._get_core(approach)
        pre_gen_tokens, post_gen_tokens, target_tokens = core.pipeline(source, target, epoch_num=epoch_num)
        
        return RepairResult(
            pre_gen_tokens=pre_gen_tokens,
            post_gen_tokens=post_gen_tokens,
            target_tokens=target_tokens
        )
    
    def get_supported_approaches(self) -> List[str]:
        """Get list of supported repair approaches."""
        return ['mint', 'me-sgd', 'me-iter', 'me-batch']

