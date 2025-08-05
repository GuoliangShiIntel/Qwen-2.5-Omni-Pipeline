# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of CDPruner for vision token pruning
"""

import torch
from cdpruner import CDPruner, Config


def example_usage():
    """Example demonstrating how to use CDPruner"""
    
    # Create configuration
    config = Config(
        num_visual_tokens=256,     # Target number of tokens after pruning
        relevance_weight=0.5,     # Balance between relevance and diversity
        enable_pruning=True,      # Enable pruning functionality
        device="CPU",             # Device for computation
        debug_mode=True           # Enable detailed logging
    )
    
    # Initialize CDPruner
    pruner = CDPruner(config)
    
    # Create example tensors
    batch_size = 7
    num_visual_tokens = 512  # Original number of visual tokens
    feature_dim = 3584        # Feature dimension
    num_text_tokens = 100     # Number of text tokens
    
    # Simulate visual features [B, N, D]
    visual_features = torch.randn(batch_size, num_visual_tokens, feature_dim)
    
    # Simulate text features [M, D]
    text_features = torch.randn(num_text_tokens, feature_dim)
    
    print("=== CDPruner Example (Simplified) ===")
    print(f"Input visual features shape: {visual_features.shape}")
    print(f"Input text features shape: {text_features.shape}")
    print(f"Target tokens after pruning: {config.num_visual_tokens}")
    
    # Recommended method: Complete all pruning operations in one call
    print("\n>>> Using simplified prune_tokens method <<<")
    pruned_features, selected_tokens, pruning_mask = pruner.prune_tokens(visual_features, text_features)
    
    print(f"Results:")
    print(f"  Pruned features shape: {pruned_features.shape}")
    print(f"  Selected token indices: {selected_tokens}")
    print(f"  Pruning mask length: {len(pruning_mask)}")
    print(f"  Number of selected tokens: {sum(pruning_mask)}")
    
    # Get statistics
    stats = pruner.get_last_pruning_statistics()
    print(f"\nPruning Statistics:")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Selected tokens: {stats.selected_tokens}")
    print(f"  Pruning ratio: {stats.pruning_ratio:.3f}")
    print(f"  Batch size: {stats.batch_size}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    example_usage()
