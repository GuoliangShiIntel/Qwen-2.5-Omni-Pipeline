# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Main CDPruner class that integrates all components
"""

import torch
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .cdpruner_config import Config
from .relevance_calculator import RelevanceCalculator
from .kernel_builder import ConditionalKernelBuilder
from .fast_dpp import FastGreedyDPP


@dataclass
class PruningStatistics:
    """Statistics about the pruning operation"""
    total_tokens: int = 0          # Total number of visual tokens before pruning
    selected_tokens: int = 0       # Number of tokens selected after pruning
    pruning_ratio: float = 0.0     # Ratio of tokens pruned (0-1)
    batch_size: int = 0            # Batch size processed


class CDPruner:
    """
    Main CDPruner class that integrates all components
    
    This class provides the complete CDPruner functionality by integrating:
    - RelevanceCalculator: Computes visual-text relevance scores
    - ConditionalKernelBuilder: Builds conditional kernel matrices
    - FastGreedyDPP: Performs diverse token selection
    
    The complete pipeline follows these steps:
    1. Compute relevance scores between visual and text features
    2. Build conditional kernel matrix combining similarity and relevance
    3. Use fast greedy DPP to select diverse and relevant tokens
    
    Usage example:
    ```python
    config = Config()
    config.num_visual_tokens = 64
    config.enable_pruning = True
    
    pruner = CDPruner(config)
    selected_tokens = pruner.select_tokens(visual_features, text_features)
    pruned_features = pruner.apply_pruning(visual_features, text_features)
    ```
    """
    
    def __init__(self, config: Config):
        """
        Constructor
        
        Args:
            config: Configuration for CDPruner
        """
        self.config = config
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self.relevance_calc = RelevanceCalculator(config)
        self.kernel_builder = ConditionalKernelBuilder(config)
        self.dpp_selector = FastGreedyDPP(config)
        
        # Statistics tracking
        self.last_statistics = PruningStatistics()
        
        if self.config.debug_mode:
            print("CDPruner initialized with configuration:")
            print(f"  num_visual_tokens: {self.config.num_visual_tokens}")
            print(f"  relevance_weight: {self.config.relevance_weight}")
            print(f"  enable_pruning: {self.config.enable_pruning}")
            print(f"  device: {self.config.device}")
    
    def select_tokens(self, visual_features: torch.Tensor, 
                     text_features: torch.Tensor) -> List[List[int]]:
        """
        Select diverse and relevant visual tokens
        
        Args:
            visual_features: Input visual features [B, N, D]
            text_features: Input text features [M, D]
            
        Returns:
            Selected token indices for each batch [B, T]
        """
        # Input validation
        if not self.config.enable_pruning:
            # If pruning is disabled, return all tokens
            return self._create_all_tokens_selection(visual_features)
        
        self._validate_input_tensors(visual_features, text_features)
        
        # Performance timing setup
        overall_start = time.perf_counter()
        
        # Get input dimensions for context
        batch_size, num_tokens, feature_dim = visual_features.shape
        num_text_tokens, text_feature_dim = text_features.shape
        
        try:
            # Step 1: Compute relevance scores
            if self.config.debug_mode:
                print("Step 1: Computing relevance scores...")
            relevance_start = time.perf_counter()
            relevance_scores = self.relevance_calc.compute(visual_features, text_features)
            relevance_end = time.perf_counter()
            relevance_duration = (relevance_end - relevance_start) * 1e6  # Convert to microseconds
            
            if self.config.debug_mode:
                print(f"  Relevance computation took: {relevance_duration:.0f} us")
            
            # Step 2: Build conditional kernel matrix
            if self.config.debug_mode:
                print("Step 2: Building conditional kernel matrix...")
            kernel_start = time.perf_counter()
            kernel_matrix = self.kernel_builder.build(visual_features, relevance_scores)
            kernel_end = time.perf_counter()
            kernel_duration = (kernel_end - kernel_start) * 1e6
            
            if self.config.debug_mode:
                print(f"  Kernel matrix construction took: {kernel_duration:.0f} us")
            
            # Step 3: Select tokens using fast greedy DPP
            if self.config.debug_mode:
                print("Step 3: Selecting tokens using DPP...")
            dpp_start = time.perf_counter()
            selected_tokens = self.dpp_selector.select(kernel_matrix, self.config.num_visual_tokens)
            dpp_end = time.perf_counter()
            dpp_duration = (dpp_end - dpp_start) * 1e6
            
            if self.config.debug_mode:
                print(f"  DPP selection took: {dpp_duration:.0f} us")
            
            # Overall timing summary
            overall_end = time.perf_counter()
            total_duration = (overall_end - overall_start) * 1e6
            
            print("\n==== Performance Summary ====")
            print(f"Total processing time: {total_duration:.0f} us ({total_duration/1000:.1f} ms)")
            
            # Component timing breakdown
            print("\nComponent Breakdown:")
            print(f"  Relevance computation: {relevance_duration:.0f} us "
                  f"({relevance_duration/total_duration*100:.1f}%)")
            print(f"  Kernel matrix build:   {kernel_duration:.0f} us "
                  f"({kernel_duration/total_duration*100:.1f}%)")
            print(f"  DPP token selection:   {dpp_duration:.0f} us "
                  f"({dpp_duration/total_duration*100:.1f}%)")
            
            # Performance metrics
            total_input_tokens = batch_size * num_tokens
            total_output_tokens = batch_size * self.config.num_visual_tokens
            print("\nPerformance Metrics:")
            print(f"  Overall throughput: {total_input_tokens / total_duration * 1e6:.0f} input tokens/sec")
            print(f"  Pruning efficiency: {total_output_tokens / total_duration * 1e6:.0f} output tokens/sec")
            print(f"  Pruning ratio: {(1.0 - self.config.num_visual_tokens / num_tokens) * 100:.1f}%")
            
            if self.config.debug_mode:
                print(f"CDPruner total processing time: {total_duration:.0f} us")
                self._print_selection_statistics(visual_features, selected_tokens)
            
            print("================================\n")
            
            return selected_tokens
            
        except Exception as e:
            raise RuntimeError(f"CDPruner.select_tokens failed: {str(e)}")
    
    def create_pruning_mask(self, selected_tokens: List[List[int]], 
                          total_tokens: int) -> List[bool]:
        """
        Create pruning mask for selected tokens
        
        Args:
            selected_tokens: Selected token indices for each batch [B, T]
            total_tokens: Total number of tokens per batch
            
        Returns:
            Boolean mask [B*N] where True indicates selected tokens
        """
        return FastGreedyDPP.create_mask(selected_tokens, total_tokens)
    
    def apply_pruning_with_selection(self, visual_features: torch.Tensor, 
                                   selected_tokens: List[List[int]]) -> torch.Tensor:
        """
        Apply pruning using pre-selected token indices
        
        Args:
            visual_features: Input visual features [B, N, D]
            selected_tokens: Pre-selected token indices [B, T]
            
        Returns:
            Pruned visual features [B, T, D] where T is num_visual_tokens
        """
        batch_size, total_tokens, feature_dim = visual_features.shape
        
        # Create output tensor with selected tokens only
        pruned_features = torch.zeros(batch_size, self.config.num_visual_tokens, feature_dim,
                                    dtype=visual_features.dtype, device=visual_features.device)
        
        for b in range(batch_size):
            batch_selected = selected_tokens[b]
            
            for t, src_token_idx in enumerate(batch_selected):
                # Copy features for this token
                pruned_features[b, t] = visual_features[b, src_token_idx]
        
        return pruned_features
    
    def prune_tokens(self, visual_features: torch.Tensor, 
                    text_features: torch.Tensor) -> tuple[torch.Tensor, List[List[int]], List[bool]]:
        """
        Complete all pruning operations in one call, returning pruned features, selected token indices and mask
        
        Args:
            visual_features: Input visual features [B, N, D]
            text_features: Input text features [M, D]
            
        Returns:
            Tuple containing:
            - pruned_features: Pruned visual features [B, T, D] 
            - selected_tokens: Selected token indices [B, T]
            - pruning_mask: Boolean mask [B*N] where True indicates selected tokens
        """
        batch_size, total_tokens, feature_dim = visual_features.shape
        num_text_tokens, text_feature_dim = text_features.shape
        
        if self.config.debug_mode:
            print("\n==== CDPruner Simplified Pruning ====")
            print("Input Information:")
            print(f"  Vision tokens (before pruning): {total_tokens}")
            print(f"  Text tokens: {num_text_tokens}")
            print(f"  Batch size: {batch_size}")
            print(f"  Target vision tokens (after pruning): {self.config.num_visual_tokens}")
        
        # Step 1: Select tokens (only called once)
        selected_tokens = self.select_tokens(visual_features, text_features)
        
        # Step 2: Apply pruning
        pruned_features = self.apply_pruning_with_selection(visual_features, selected_tokens)
        
        # Step 3: Create mask
        pruning_mask = self.create_pruning_mask(selected_tokens, total_tokens)
        
        if self.config.debug_mode:
            pruning_ratio = 1.0 - self.config.num_visual_tokens / total_tokens
            reduction_percentage = pruning_ratio * 100.0
            print(f"\nPruning completed: {total_tokens} -> {self.config.num_visual_tokens} tokens ({reduction_percentage:.1f}% reduction)")
            print("======================================\n")
        
        return pruned_features, selected_tokens, pruning_mask
    
    def compute_pruning_ratio(self) -> float:
        """
        Compute current pruning ratio
        
        Returns:
            Ratio of selected tokens to default token count
        """
        return self.config.num_visual_tokens / self.get_default_token_count()
    
    def get_default_token_count(self) -> int:
        """
        Get default token count for the model
        
        Returns:
            Default number of visual tokens (e.g., 576 for LLaVA)
        """
        # LLaVA typical token count (can be made configurable)
        return 576  # 24x24 patches for most LLaVA configurations
    
    def get_last_pruning_statistics(self) -> PruningStatistics:
        """
        Get statistics from the last pruning operation
        
        Returns:
            Pruning statistics
        """
        return self.last_statistics
    
    def get_config(self) -> Config:
        """
        Get current configuration
        
        Returns:
            Current configuration
        """
        return self.config
    
    def _validate_input_tensors(self, visual_features: torch.Tensor, 
                              text_features: torch.Tensor) -> None:
        """
        Validate input tensor shapes and types
        
        Args:
            visual_features: Visual features tensor
            text_features: Text features tensor
        """
        # Validate visual features
        if visual_features.dim() != 3:
            raise ValueError("Visual features must be 3D tensor [B, N, D]")
        
        # Validate text features
        if text_features.dim() != 2:
            raise ValueError("Text features must be 2D tensor [M, D]")
        
        batch_size, num_tokens, visual_dim = visual_features.shape
        num_text_tokens, text_dim = text_features.shape
        
        # Check feature dimension consistency
        if visual_dim != text_dim:
            raise ValueError("Visual and text features must have same feature dimension")
        
        # Check if we can select the requested number of tokens
        if self.config.num_visual_tokens > num_tokens:
            raise ValueError("Cannot select more tokens than available in visual features")
        
        # Check tensor data types
        if not (visual_features.dtype == torch.float32 and text_features.dtype == torch.float32):
            raise ValueError("Input tensors must be float32 type")
    
    def _create_all_tokens_selection(self, visual_features: torch.Tensor) -> List[List[int]]:
        """
        Create selection that includes all tokens (when pruning is disabled)
        
        Args:
            visual_features: Visual features tensor
            
        Returns:
            All token indices for each batch
        """
        batch_size, total_tokens, _ = visual_features.shape
        
        all_tokens = []
        for b in range(batch_size):
            batch_tokens = list(range(total_tokens))
            all_tokens.append(batch_tokens)
        
        return all_tokens
    
    def _print_selection_statistics(self, visual_features: torch.Tensor, 
                                  selected_tokens: List[List[int]]) -> None:
        """
        Print detailed selection statistics (debug mode)
        
        Args:
            visual_features: Visual features tensor
            selected_tokens: Selected token indices
        """
        batch_size, total_tokens, _ = visual_features.shape
        
        print("Selection Statistics:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Selected tokens: {self.config.num_visual_tokens}")
        print(f"  Pruning ratio: {(1.0 - self.config.num_visual_tokens / total_tokens) * 100:.1f}%")
        
        for b in range(min(batch_size, 3)):  # Show first 3 batches max
            print(f"  Batch {b} selected indices: [", end="")
            batch_tokens = selected_tokens[b]
            for i in range(min(len(batch_tokens), 10)):  # Show first 10 indices
                if i > 0:
                    print(", ", end="")
                print(batch_tokens[i], end="")
            if len(batch_tokens) > 10:
                print(", ...", end="")
            print("]")
        
        # Update statistics
        self.last_statistics.total_tokens = total_tokens
        self.last_statistics.selected_tokens = self.config.num_visual_tokens
        self.last_statistics.pruning_ratio = 1.0 - self.config.num_visual_tokens / total_tokens
        self.last_statistics.batch_size = batch_size
