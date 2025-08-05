# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Fast greedy DPP (Determinantal Point Process) algorithm for token selection
"""

import torch
import numpy as np
from typing import List, Tuple
from .cdpruner_config import Config


class FastGreedyDPP:
    """
    Fast greedy DPP (Determinantal Point Process) algorithm for token selection
    
    This class implements the fast greedy approximation algorithm for maximizing
    the determinant of a subset selection from a kernel matrix. The algorithm
    is based on the CDPruner paper and provides O(T²N) complexity where T is
    the number of tokens to select and N is the total number of tokens.
    
    The core algorithm follows these steps:
    1. Initialize diagonal scores (marginal gains)
    2. Greedily select tokens with maximum marginal gain
    3. Update orthogonalized vectors using Gram-Schmidt process
    4. Update marginal gains by subtracting orthogonal projections
    """
    
    def __init__(self, config: Config):
        """
        Constructor
        
        Args:
            config: Configuration for the DPP selector
        """
        self.config = config
    
    def select(self, kernel: torch.Tensor, num_tokens: int) -> List[List[int]]:
        """
        Select diverse tokens using fast greedy DPP algorithm
        
        Args:
            kernel: Conditional kernel matrix [B, N, N]
            num_tokens: Number of tokens to select
            
        Returns:
            Selected token indices for each batch [B, T]
        """
        # Input validation
        if kernel.dim() != 3:
            raise ValueError("Kernel must be 3D tensor [B, N, N]")
        
        batch_size, n_tokens_1, n_tokens_2 = kernel.shape
        
        if n_tokens_1 != n_tokens_2:
            raise ValueError("Kernel matrix must be square [B, N, N]")
        
        if num_tokens > n_tokens_1:
            raise ValueError("Cannot select more tokens than available")
        
        batch_results = []
        
        # Process each batch independently
        for b in range(batch_size):
            batch_result = self.select_single_batch(kernel[b], num_tokens)
            batch_results.append(batch_result)
        
        return batch_results
    
    def select_single_batch(self, kernel: torch.Tensor, num_tokens: int) -> List[int]:
        """
        Select tokens for a single batch
        
        Args:
            kernel: Kernel matrix [N, N] for single batch
            num_tokens: Number of tokens to select
            
        Returns:
            Selected token indices for this batch
        """
        total_tokens = kernel.shape[0]
        
        # Initialize working tensors for this batch
        # cis: Orthogonalized vectors [T, N] where T is the number of selected tokens
        cis = torch.zeros(num_tokens, total_tokens, dtype=kernel.dtype, device=kernel.device)
        
        # di2s: Diagonal elements (marginal gains) [N]
        di2s = torch.diag(kernel).clone()
        
        selected_indices = []
        
        # Greedy selection loop - this is the core DPP algorithm
        for t in range(num_tokens):
            # Find the token with maximum marginal gain
            best_idx = self.argmax(di2s)
            selected_indices.append(best_idx.item())
            
            # Compute the new orthogonalized vector e_i
            # eis = (kernel[best_idx] - sum(cis[:t] * cis[:t, best_idx])) / sqrt(di2s[best_idx])
            self.update_orthogonal_vector(kernel, best_idx, t, cis, di2s)
            
            # Update marginal gains by subtracting the squared new orthogonal vector
            # di2s -= square(eis)
            self.update_marginal_gains(t, best_idx, cis, di2s)
            
            # Set the selected token's gain to negative infinity to prevent re-selection
            di2s[best_idx] = float('-inf')
        
        # Sort the selected indices for deterministic output
        selected_indices.sort()
        
        return selected_indices
    
    def argmax(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Find index with maximum value
        
        Args:
            scores: Score tensor [N]
            
        Returns:
            Index of maximum value
        """
        if scores.numel() == 0:
            raise ValueError("Cannot find argmax of empty tensor")
        
        return torch.argmax(scores)
    
    def update_orthogonal_vector(self, kernel: torch.Tensor, selected_idx: torch.Tensor, 
                               iteration: int, cis: torch.Tensor, di2s: torch.Tensor) -> None:
        """
        Update orthogonal vector using Gram-Schmidt process
        
        Args:
            kernel: Kernel matrix [N, N]
            selected_idx: Newly selected token index
            iteration: Current iteration (number of previously selected tokens)
            cis: Orthogonalized vectors [T, N]
            di2s: Current diagonal scores [N]
        """
        # This implements the key DPP orthogonalization step:
        # eis = (kernel[selected_idx] - sum(cis[:iteration] * cis[:iteration, selected_idx])) / sqrt(di2s[selected_idx])
        
        total_tokens = kernel.shape[0]
        
        # Get the normalization factor
        norm_factor = torch.sqrt(di2s[selected_idx] + self.config.numerical_threshold)
        
        # Get kernel row for selected token
        kernel_row = kernel[selected_idx]  # [N]
        
        # Subtract the projection onto previously selected vectors
        # sum(cis[:iteration, selected_idx] * cis[:iteration, j]) for each j
        if iteration > 0:
            # Get the projections onto previously selected vectors
            prev_cis = cis[:iteration]  # [iteration, N]
            prev_cis_selected = prev_cis[:, selected_idx]  # [iteration]
            
            # Compute projection: sum(cis[:iteration, selected_idx] * cis[:iteration, :])
            projection = torch.sum(prev_cis_selected.unsqueeze(1) * prev_cis, dim=0)  # [N]
        else:
            projection = torch.zeros_like(kernel_row)
        
        # Store the orthogonalized vector element
        cis[iteration] = (kernel_row - projection) / norm_factor
    
    def update_marginal_gains(self, iteration: int, selected_idx: torch.Tensor, 
                            cis: torch.Tensor, di2s: torch.Tensor) -> None:
        """
        Update marginal gains after selecting a token
        
        Args:
            iteration: Current iteration
            selected_idx: Newly selected token index
            cis: Orthogonalized vectors [T, N]
            di2s: Diagonal scores to update [N]
        """
        # This implements: di2s -= square(eis)
        # where eis is the newly computed orthogonal vector cis[iteration, :]
        
        # Get the newly computed orthogonal vector
        eis = cis[iteration]  # [N]
        
        # Update marginal gains for all tokens
        di2s -= eis * eis
    
    @staticmethod
    def create_mask(selected_indices: List[List[int]], total_tokens: int) -> List[bool]:
        """
        Create boolean mask from selected indices
        
        Args:
            selected_indices: Selected indices for each batch [B, T]
            total_tokens: Total number of tokens
            
        Returns:
            Boolean mask [B*N] where True indicates selected tokens
        """
        if not selected_indices:
            return [False] * total_tokens
        
        batch_size = len(selected_indices)
        mask = [False] * (batch_size * total_tokens)
        
        for b in range(batch_size):
            for idx in selected_indices[b]:
                if idx < total_tokens:
                    mask[b * total_tokens + idx] = True
        
        return mask
    
    @staticmethod
    def compute_determinant_approximation(kernel: torch.Tensor, 
                                        selected_indices: List[int]) -> float:
        """
        Compute approximate determinant for validation
        
        Args:
            kernel: Kernel matrix [N, N] (single batch only)
            selected_indices: Selected token indices
            
        Returns:
            Approximated determinant value
        """
        # This is a simplified approximation for validation purposes
        # In practice, the greedy algorithm approximates the determinant maximization
        
        if not selected_indices:
            return 0.0
        
        if kernel.dim() != 2:
            raise ValueError("Determinant approximation only supports 2D kernel matrix")
        
        # Compute the product of diagonal elements of selected tokens as approximation
        det_approx = 1.0
        for idx in selected_indices:
            det_approx *= kernel[idx, idx].item()
        
        return det_approx
