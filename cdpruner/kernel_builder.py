# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Builder for conditional kernel matrices used in DPP-based token selection
"""

import torch
import torch.nn.functional as F
import time
from typing import Union
from .cdpruner_config import Config


class ConditionalKernelBuilder:
    """
    Builder for conditional kernel matrices used in DPP-based token selection
    
    This class implements the conditional kernel matrix construction that combines
    visual feature similarity with relevance-based weighting as described in the
    CDPruner paper. The kernel matrix is computed as:
    
    L̃ = diag(r) · L · diag(r)
    
    where L is the similarity matrix and r is the relevance vector.
    """
    
    def __init__(self, config: Config):
        """
        Constructor
        
        Args:
            config: Configuration for the kernel builder
        """
        self.config = config
    
    def build(self, visual_features: torch.Tensor, relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Build conditional kernel matrix L̃ = diag(r) · L · diag(r)
        
        Args:
            visual_features: Visual feature embeddings [B, N, D]
            relevance_scores: Relevance scores [B, N]
            
        Returns:
            Conditional kernel matrix [B, N, N]
        """
        # Input validation
        if visual_features.dim() != 3:
            raise ValueError("Visual features must be 3D tensor [B, N, D]")
        if relevance_scores.dim() != 2:
            raise ValueError("Relevance scores must be 2D tensor [B, N]")
        
        batch_size, num_tokens, feature_dim = visual_features.shape
        relevance_batch_size, relevance_num_tokens = relevance_scores.shape
        
        # Check shape consistency
        if relevance_batch_size != batch_size or relevance_num_tokens != num_tokens:
            raise ValueError("Visual features and relevance scores must have consistent batch size and token count")
        
        # Performance timing for kernel building steps
        kernel_build_start = time.perf_counter()
        
        if self.config.debug_mode:
            print(f"\n==== Kernel Build Performance Analysis ====")
            print(f"Input tensors: visual_features[{batch_size}, {num_tokens}, {feature_dim}], "
                  f"relevance_scores[{batch_size}, {num_tokens}]")
        
        # Step 1: L2 normalize visual features along the last dimension
        # This is equivalent to: image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
        normalize_start = time.perf_counter()
        normalized_features = self.l2_normalize_features(visual_features)
        normalize_end = time.perf_counter()
        normalize_duration = (normalize_end - normalize_start) * 1e6  # Convert to microseconds
        
        # Step 2: Compute similarity matrix L = normalized_features @ normalized_features.T
        # This gives us the base similarity matrix [B, N, N]
        similarity_start = time.perf_counter()
        if self.config.device.upper() == "GPU" and torch.cuda.is_available():
            similarity_matrix = self.compute_similarity_matrix_gpu(normalized_features)
        else:
            similarity_matrix = self.compute_similarity_matrix(normalized_features)
        similarity_end = time.perf_counter()
        similarity_duration = (similarity_end - similarity_start) * 1e6
        
        # Step 3: Build conditional kernel matrix L̃ = diag(r) · L · diag(r)
        # Following CDPruner: kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
        conditional_start = time.perf_counter()
        conditional_kernel = self.build_conditional_kernel(similarity_matrix, relevance_scores)
        conditional_end = time.perf_counter()
        conditional_duration = (conditional_end - conditional_start) * 1e6
        
        kernel_build_end = time.perf_counter()
        total_kernel_duration = (kernel_build_end - kernel_build_start) * 1e6
        
        # Print performance breakdown
        if self.config.debug_mode:
            print(f"Kernel Build Breakdown:")
            print(f"  L2 normalization [{batch_size}, {num_tokens}, {feature_dim}]: "
                  f"{normalize_duration:.0f} us ({normalize_duration/total_kernel_duration*100:.1f}%)")
            print(f"  Similarity matrix [{batch_size}, {num_tokens}, {num_tokens}]: "
                  f"{similarity_duration:.0f} us ({similarity_duration/total_kernel_duration*100:.1f}%)")
            print(f"  Conditional kernel [{batch_size}, {num_tokens}, {num_tokens}]: "
                  f"{conditional_duration:.0f} us ({conditional_duration/total_kernel_duration*100:.1f}%)")
            
            print(f"Total kernel build time: {total_kernel_duration:.0f} us ({total_kernel_duration/1000:.1f} ms)")
            
            # Performance metrics
            total_operations = batch_size * num_tokens * num_tokens  # Dominant operation is N^2
            print(f"Kernel build throughput: {total_operations / total_kernel_duration * 1e6:.0f} ops/sec")
            print("==========================================\n")
        
        return conditional_kernel
    
    def compute_similarity_matrix_gpu(self, features: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated similarity matrix computation using PyTorch
        
        Args:
            features: Normalized visual features [B, N, D]
            
        Returns:
            Similarity matrix [B, N, N]
        """
        # features: [B, N, D] - normalized visual features
        # Result: [B, N, N] - similarity matrix
        
        try:
            # Move to GPU if available
            if torch.cuda.is_available() and self.config.device.upper() == "GPU":
                features = features.cuda()
            
            # Batch matrix multiplication: [B, N, D] @ [B, D, N] = [B, N, N]
            similarity_matrix = torch.matmul(features, features.transpose(-2, -1))
            
            return similarity_matrix
            
        except Exception as e:
            # Fallback to CPU implementation if GPU fails
            if self.config.debug_mode:
                print(f"GPU MatMul failed, falling back to CPU: {e}")
            return self.compute_similarity_matrix(features)
    
    def compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between visual features
        
        Args:
            features: Visual feature embeddings [B, N, D]
            
        Returns:
            Similarity matrix [B, N, N]
        """
        # features: [B, N, D] - normalized visual features
        # Result: [B, N, N] - similarity matrix
        
        # Use PyTorch's batch matrix multiplication
        # [B, N, D] @ [B, D, N] = [B, N, N]
        similarity_matrix = torch.matmul(features, features.transpose(-2, -1))
        
        return similarity_matrix
    
    def l2_normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize features along the last dimension
        
        Args:
            features: Input features [B, N, D]
            
        Returns:
            Normalized features [B, N, D]
        """
        # Add small epsilon for numerical stability
        norm = torch.norm(features, p=2, dim=-1, keepdim=True) + self.config.numerical_threshold
        return features / norm
    
    def build_conditional_kernel(self, similarity_matrix: torch.Tensor, 
                               relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Build conditional kernel matrix using relevance weighting
        
        Args:
            similarity_matrix: Base similarity matrix [B, N, N]
            relevance_scores: Token relevance scores [B, N]
            
        Returns:
            Conditional kernel matrix [B, N, N]
        """
        # similarity_matrix: [B, N, N]
        # relevance_scores: [B, N]
        # Result: [B, N, N] - conditional kernel matrix
        
        # Implementation of: kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
        # This is equivalent to: kernel[b, i, j] = relevance[b, i] * similarity[b, i, j] * relevance[b, j]
        
        # Expand relevance scores for broadcasting
        # relevance.unsqueeze(2): [B, N, 1]
        # relevance.unsqueeze(1): [B, 1, N]
        relevance_i = relevance_scores.unsqueeze(2)  # [B, N, 1]
        relevance_j = relevance_scores.unsqueeze(1)  # [B, 1, N]
        
        # Apply conditional weighting: r[i] * similarity[i,j] * r[j]
        conditional_kernel = relevance_i * similarity_matrix * relevance_j
        
        return conditional_kernel
