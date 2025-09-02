"""
Configuration for ultra-tiny model experiments testing emergent vocabulary development.

This module defines experimental configurations for testing the hypothesis that
ultra-constrained models (1M-10M parameters) develop more efficient emergent
vocabularies under information bottleneck constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for ultra-tiny transformer models."""
    
    # Model architecture
    vocab_size: int = 1000  # Small vocabulary for ultra-constrained setting
    max_seq_len: int = 128  # Short sequences to enforce compression
    d_model: int = 64      # Tiny embedding dimension
    n_heads: int = 2       # Minimal attention heads
    n_layers: int = 2      # Very shallow
    d_ff: int = 128        # Small feed-forward
    dropout: float = 0.1
    
    # Parameter count targeting
    target_params: int = 1_000_000  # 1M parameters
    
    @property
    def estimated_params(self) -> int:
        """Estimate parameter count for this configuration."""
        # Embedding parameters
        embedding_params = self.vocab_size * self.d_model
        
        # Transformer layer parameters (simplified estimation)
        attention_params = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        ffn_params = 2 * self.d_model * self.d_ff + self.d_ff + self.d_model
        layer_params = attention_params + ffn_params
        
        # Total parameters
        total_params = embedding_params + (layer_params * self.n_layers)
        return total_params
    
    def scale_to_target(self) -> 'ModelConfig':
        """Automatically scale model to hit target parameter count."""
        current_params = self.estimated_params
        scale_factor = (self.target_params / current_params) ** 0.5
        
        # Scale d_model primarily
        new_d_model = max(16, int(self.d_model * scale_factor))
        new_d_ff = max(32, int(self.d_ff * scale_factor))
        
        return ModelConfig(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=new_d_model,
            n_heads=min(self.n_heads, new_d_model // 16),
            n_layers=self.n_layers,
            d_ff=new_d_ff,
            dropout=self.dropout,
            target_params=self.target_params
        )


@dataclass
class TaskConfig:
    """Configuration for communication tasks."""
    
    # Task type
    task_type: str = "referential_game"  # or "compositional_generation"
    
    # Environment setup
    world_size: Tuple[int, int] = (8, 8)  # Grid world dimensions
    n_objects: int = 16  # Number of objects in world
    n_properties: int = 4   # Properties per object (color, shape, etc.)
    property_values: int = 4  # Values per property
    
    # Communication constraints
    max_message_length: int = 8   # Ultra-constrained messages
    channel_noise: float = 0.0    # Communication noise level
    bandwidth_limit: int = 32     # Bits per message maximum
    
    # Task difficulty progression
    difficulty_levels: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"n_objects": 4, "n_properties": 2, "max_msg_len": 4},
        {"n_objects": 8, "n_properties": 3, "max_msg_len": 6},
        {"n_objects": 16, "n_properties": 4, "max_msg_len": 8},
        {"n_objects": 32, "n_properties": 5, "max_msg_len": 12},
    ])


@dataclass
class TrainingConfig:
    """Training configuration for communication experiments."""
    
    # Training dynamics
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 1000
    patience: int = 50
    
    # Communication-specific training
    temperature_schedule: List[Tuple[int, float]] = field(default_factory=lambda: [
        (0, 5.0),      # High temperature for exploration
        (100, 2.0),    # Cool down for exploitation
        (300, 1.0),    # Fine-tune
        (500, 0.5),    # Sharp protocols
    ])
    
    # Curriculum learning
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"epochs": 200, "task_difficulty": 0, "vocab_constraint": 0.8},
        {"epochs": 300, "task_difficulty": 1, "vocab_constraint": 0.6},
        {"epochs": 300, "task_difficulty": 2, "vocab_constraint": 0.4},
        {"epochs": 200, "task_difficulty": 3, "vocab_constraint": 0.2},
    ])
    
    # Regularization for emergent vocabulary
    entropy_regularization: float = 0.01  # Encourage diverse vocabulary
    mutual_info_target: float = 2.0       # Target mutual information
    compression_penalty: float = 0.1      # Penalty for long messages


@dataclass 
class ExperimentConfig:
    """Master configuration for vocabulary emergence experiments."""
    
    # Experiment metadata
    experiment_name: str = "ultra_constrained_vocab_emergence"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameter sweep
    parameter_counts: List[int] = field(default_factory=lambda: [
        1_000_000,    # 1M parameters
        2_000_000,    # 2M parameters  
        5_000_000,    # 5M parameters
        10_000_000,   # 10M parameters
        25_000_000,   # 25M parameters
        50_000_000,   # 50M parameters
        100_000_000,  # 100M parameters
        500_000_000,  # 500M parameters (upper bound)
    ])
    
    # Bottleneck constraints sweep
    bottleneck_strengths: List[float] = field(default_factory=lambda: [
        0.1,   # Very tight bottleneck
        0.3,   # Tight bottleneck
        0.5,   # Medium bottleneck
        0.7,   # Loose bottleneck
        1.0,   # No bottleneck (control)
    ])
    
    # Component configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    task_config: TaskConfig = field(default_factory=TaskConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 100
    save_vocabulary_snapshots: bool = True
    track_compositionality: bool = True
    
    # Analysis configuration
    vocab_analysis_intervals: List[int] = field(default_factory=lambda: [
        10, 25, 50, 100, 200, 500, 1000
    ])
    
    compositionality_tests: List[str] = field(default_factory=lambda: [
        "systematic_generalization",
        "constituent_swap", 
        "novel_combinations",
        "hierarchical_structure"
    ])
    
    def get_model_configs(self) -> List[ModelConfig]:
        """Generate model configurations for parameter sweep."""
        configs = []
        for param_count in self.parameter_counts:
            config = self.model_config.copy()
            config.target_params = param_count
            configs.append(config.scale_to_target())
        return configs
    
    def get_experiment_variants(self) -> List[Dict[str, Any]]:
        """Generate all experimental variants."""
        variants = []
        
        for param_count in self.parameter_counts:
            for bottleneck in self.bottleneck_strengths:
                variant = {
                    "param_count": param_count,
                    "bottleneck_strength": bottleneck,
                    "model_config": self.model_config.copy(),
                    "task_config": self.task_config.copy(),
                    "training_config": self.training_config.copy(),
                }
                # Adjust model config for this parameter count
                variant["model_config"].target_params = param_count
                variant["model_config"] = variant["model_config"].scale_to_target()
                
                # Adjust bottleneck in task config
                variant["task_config"].max_message_length = max(
                    1, int(self.task_config.max_message_length * bottleneck)
                )
                
                variants.append(variant)
        
        return variants


# Predefined experiment configurations
ULTRA_TINY_CONFIG = ExperimentConfig(
    experiment_name="ultra_tiny_vocab_emergence",
    parameter_counts=[500_000, 1_000_000, 2_000_000],
    bottleneck_strengths=[0.1, 0.3, 0.5],
    model_config=ModelConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        vocab_size=500,
        max_seq_len=64
    )
)

SCALING_STUDY_CONFIG = ExperimentConfig(
    experiment_name="parameter_scaling_study",
    parameter_counts=[1_000_000, 5_000_000, 25_000_000, 100_000_000, 500_000_000],
    bottleneck_strengths=[0.3],  # Fixed bottleneck
    model_config=ModelConfig(
        d_model=64,
        n_heads=4, 
        n_layers=3,
        vocab_size=1000,
        max_seq_len=128
    )
)

BOTTLENECK_STUDY_CONFIG = ExperimentConfig(
    experiment_name="bottleneck_strength_study", 
    parameter_counts=[5_000_000],  # Fixed model size
    bottleneck_strengths=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    model_config=ModelConfig(
        d_model=48,
        n_heads=3,
        n_layers=2,
        vocab_size=750,
        max_seq_len=96
    )
)


def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate experimental configuration and return any issues."""
    issues = []
    
    # Check model parameter estimates
    for param_count in config.parameter_counts:
        test_config = config.model_config.copy()
        test_config.target_params = param_count
        scaled_config = test_config.scale_to_target()
        estimated = scaled_config.estimated_params
        
        if abs(estimated - param_count) / param_count > 0.1:
            issues.append(f"Parameter count {param_count} estimate off by >10%: {estimated}")
    
    # Check bottleneck constraints
    if max(config.bottleneck_strengths) > 1.0:
        issues.append("Bottleneck strengths should not exceed 1.0")
    
    # Check vocabulary size constraints
    if config.model_config.vocab_size < config.task_config.max_message_length:
        issues.append("Vocabulary size too small for message length")
    
    return issues