# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Configuration structure for CDPruner algorithm
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration structure for CDPruner algorithm"""
    
    # Number of visual tokens to retain after pruning
    num_visual_tokens: int = 256
    
    # Weight for balancing relevance vs diversity (0.0 to 1.0)
    relevance_weight: float = 0.5
    
    # Whether to enable pruning functionality
    enable_pruning: bool = True
    
    # Device to run CDPruner computations on
    device: str = "CPU"
    
    # Whether to enable debug output
    debug_mode: bool = False
    
    # Threshold for numerical stability
    numerical_threshold: float = 1e-6
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.num_visual_tokens <= 0:
            raise ValueError("num_visual_tokens must be greater than 0")
        
        if not (0.0 <= self.relevance_weight <= 1.0):
            raise ValueError("relevance_weight must be in range [0.0, 1.0]")
        
        if self.numerical_threshold <= 0.0:
            raise ValueError("numerical_threshold must be positive")
        
        if not self.device:
            raise ValueError("device cannot be empty")
