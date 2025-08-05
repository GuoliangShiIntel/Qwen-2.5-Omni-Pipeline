# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Relevance calculator for computing visual-text relevance scores
"""

import torch
import torch.nn.functional as F
from typing import Union
from .cdpruner_config import Config


class RelevanceCalculator:
    """Class for computing relevance scores between visual and text features"""
    
    def __init__(self, config: Config):
        """
        Constructor
        
        Args:
            config: Configuration for the calculator
        """
        self.config = config
    
    def compute(self, visual_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance scores between visual embeddings and text embeddings
        
        Args:
            visual_embeds: Visual feature embeddings [B, N, C]
            text_embeds: Text feature embeddings [M, C]
            
        Returns:
            Relevance scores tensor [B, N]
        """
        # Input validation
        if visual_embeds.dim() != 3:
            raise ValueError("Visual embeddings must be 3D tensor [B, N, C]")
        if text_embeds.dim() != 2:
            raise ValueError("Text embeddings must be 2D tensor [M, C]")
        
        batch_size, num_visual_tokens, visual_dim = visual_embeds.shape
        num_text_tokens, text_dim = text_embeds.shape
        
        # For simplicity, we assume visual and text embeddings have the same dimension
        # In practice, they might need to be projected to the same space
        if visual_dim != text_dim:
            raise ValueError("Visual and text embeddings must have the same feature dimension")
        
        # Step 1: L2 normalize visual embeddings along the last dimension
        visual_normalized = self.l2_normalize(visual_embeds)
        
        # Step 2: L2 normalize text embeddings along the last dimension  
        text_normalized = self.l2_normalize(text_embeds)
        
        # Step 3: Compute cosine similarity between visual and text embeddings
        # relevance = visual_embeds @ text_embeds.T  => [B, N, M]
        relevance_matrix = self.matrix_multiply(visual_normalized, text_normalized)
        
        # Step 4: Take negative mean across text tokens dimension to get relevance scores
        # This follows the CDPruner implementation: relevance = (-relevance).mean(dim=-1)
        relevance_scores = self.compute_negative_mean(relevance_matrix)
        
        # Step 5: Min-max normalize the relevance scores
        normalized_relevance = self.min_max_normalize(relevance_scores)
        
        return normalized_relevance
    
    def l2_normalize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize tensor along the last dimension
        
        Args:
            input_tensor: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        # Add small epsilon for numerical stability
        norm = torch.norm(input_tensor, p=2, dim=-1, keepdim=True) + self.config.numerical_threshold
        return input_tensor / norm
    
    def min_max_normalize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize tensor
        
        Args:
            input_tensor: Input tensor to normalize [B, N]
            
        Returns:
            Normalized tensor [B, N]
        """
        if input_tensor.dim() != 2:
            raise ValueError("Min-max normalization only supports 2D tensors")
        
        # For 2D tensor [B, N], normalize each batch independently
        batch_size, num_tokens = input_tensor.shape
        result = torch.zeros_like(input_tensor)
        
        for b in range(batch_size):
            batch_data = input_tensor[b]
            min_val = torch.min(batch_data)
            max_val = torch.max(batch_data)
            
            # Avoid division by zero
            range_val = max_val - min_val
            if range_val < self.config.numerical_threshold:
                range_val = 1.0  # If all values are the same, set to 1
            
            # Normalize batch b
            result[b] = (batch_data - min_val + self.config.numerical_threshold) / range_val
        
        return result
    
    def matrix_multiply(self, visual_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix multiplication between visual and text embeddings
        
        Args:
            visual_embeds: Visual embeddings [B, N, C]
            text_embeds: Text embeddings [M, C]
            
        Returns:
            Similarity matrix [B, N, M]
        """
        # visual_embeds: [B, N, C]
        # text_embeds: [M, C] 
        # Result: [B, N, M]
        
        # Transpose text embeddings: [M, C] -> [C, M]
        text_transposed = text_embeds.transpose(0, 1)  # [C, M]
        
        # Batch matrix multiplication: [B, N, C] @ [C, M] = [B, N, M]
        result = torch.matmul(visual_embeds, text_transposed)
        
        return result
    
    def compute_negative_mean(self, relevance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute negative mean across the last dimension
        
        Args:
            relevance_matrix: Input relevance matrix [B, N, M]
            
        Returns:
            Mean relevance scores [B, N]
        """
        # relevance_matrix: [B, N, M]
        # Result: [B, N] - mean across the last dimension with negation
        
        # Compute mean across text tokens (last dimension) and apply negation
        mean_relevance = torch.mean(relevance_matrix, dim=-1)  # [B, N]
        return -mean_relevance  # Apply negation as in CDPruner
