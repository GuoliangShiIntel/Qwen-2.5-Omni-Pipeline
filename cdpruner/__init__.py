# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CDPruner: Conditional Determinantal Point Processes for Visual Token Pruning

This package implements the CDPruner algorithm for pruning visual tokens in
multimodal language models while maintaining diversity and relevance.
"""

from .cdpruner import CDPruner
from .cdpruner_config import Config
from .relevance_calculator import RelevanceCalculator
from .kernel_builder import ConditionalKernelBuilder
from .fast_dpp import FastGreedyDPP

__version__ = "1.0.0"
__all__ = [
    "CDPruner",
    "Config", 
    "RelevanceCalculator",
    "ConditionalKernelBuilder",
    "FastGreedyDPP"
]
