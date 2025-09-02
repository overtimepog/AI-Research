"""
Test script to validate the experimental validation framework.

This script runs basic tests to ensure all components work together properly
and provides a quick way to verify the framework before running full experiments.
"""

import torch
import numpy as np
import random
from pathlib import Path
import json

from experiment_config import ExperimentConfig, ModelConfig, TaskConfig, TrainingConfig, ULTRA_TINY_CONFIG
from vocabulary_analysis import VocabularyAnalyzer, VocabularySnapshot
from bottleneck_experiment import BottleneckExperiment, UltraTinyTransformer, ReferentialGameEnvironment
from metrics import CommunicationMetrics, BottleneckMetrics, MessageBatch, MetricsAggregator


def test_model_creation():
    """Test ultra-tiny transformer creation and parameter counting."""
    print("Testing model creation...")
    
    config = ModelConfig(
        vocab_size=500,
        max_seq_len=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
        target_params=1_000_000
    )
    
    # Scale to target parameters
    scaled_config = config.scale_to_target()
    print(f"Scaled config: d_model={scaled_config.d_model}, estimated_params={scaled_config.estimated_params}")
    
    # Create model
    model = UltraTinyTransformer(scaled_config)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"Actual parameters: {actual_params:,}")
    
    # Test forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    print(f"Forward pass successful: {logits.shape}")
    
    # Test generation
    context_tensor = torch.zeros(1, scaled_config.d_model)
    message = model.generate(context_tensor, max_length=8)
    print(f"Generated message: {message}")
    
    print("✓ Model creation test passed\n")
    return True


def test_environment_and_metrics():
    """Test environment and metrics computation."""
    print("Testing environment and metrics...")
    
    task_config = TaskConfig(
        world_size=(4, 4),
        n_objects=8,
        n_properties=3,
        max_message_length=6
    )
    
    env = ReferentialGameEnvironment(task_config)
    
    # Generate some sample data
    messages = []
    contexts = []
    rewards = []
    
    for _ in range(50):
        context, target_idx = env.sample_task()
        # Generate random message
        message = [random.randint(0, 99) for _ in range(random.randint(2, 6))]
        reward = env.evaluate_message(message, context, target_idx)
        
        messages.append(message)
        contexts.append(context)
        rewards.append(reward)
    
    print(f"Generated {len(messages)} sample messages")
    
    # Test metrics
    batch = MessageBatch(messages=messages, contexts=contexts, rewards=rewards)
    
    comm_metrics = CommunicationMetrics(vocab_size=100, max_message_length=6)
    bottleneck_metrics = BottleneckMetrics(vocab_size=100)
    
    basic_metrics = comm_metrics.compute_basic_metrics(batch)
    entropy_metrics = comm_metrics.compute_entropy_metrics(batch)
    efficiency_metrics = comm_metrics.compute_efficiency_metrics(batch)
    compression_metrics = bottleneck_metrics.compute_compression_metrics(batch)
    
    print("Basic metrics:", basic_metrics)
    print("Entropy metrics:", entropy_metrics)
    print("Efficiency metrics:", efficiency_metrics)
    print("Compression metrics:", compression_metrics)
    
    print("✓ Environment and metrics test passed\n")
    return True


def test_vocabulary_analysis():
    """Test vocabulary analysis functionality."""
    print("Testing vocabulary analysis...")
    
    # Create sample data
    messages = [
        [1, 2, 3],
        [1, 4, 5],
        [2, 3, 6],
        [1, 2, 7],
        [4, 5, 8]
    ]
    contexts = [{"obj": i, "prop": i % 3} for i in range(len(messages))]
    rewards = [0.8, 0.6, 0.9, 0.7, 0.5]
    
    snapshot1 = VocabularySnapshot(
        step=100,
        messages=messages,
        contexts=contexts,
        rewards=rewards,
        model_params=1000000,
        bottleneck_strength=0.3
    )
    
    # Create more evolved snapshot
    evolved_messages = [
        [1, 2],     # More compressed
        [1, 4],
        [2, 3],
        [1, 2],
        [4, 5]
    ]
    evolved_rewards = [0.9, 0.8, 0.95, 0.9, 0.7]  # Better performance
    
    snapshot2 = VocabularySnapshot(
        step=500,
        messages=evolved_messages,
        contexts=contexts,
        rewards=evolved_rewards,
        model_params=1000000,
        bottleneck_strength=0.3
    )
    
    # Analyze vocabulary evolution
    analyzer = VocabularyAnalyzer(vocab_size=100, max_message_length=8)
    analyzer.add_snapshot(snapshot1)
    analyzer.add_snapshot(snapshot2)
    
    # Generate analysis report
    report = analyzer.generate_analysis_report()
    print("Analysis report keys:", list(report.keys()))
    print("Final metrics:", report["final_metrics"])
    print("Key findings:", report["key_findings"])
    
    # Test dynamics computation
    dynamics = analyzer.compute_emergence_dynamics()
    print("Emergence dynamics:", list(dynamics.keys()))
    
    print("✓ Vocabulary analysis test passed\n")
    return True


def test_experiment_config():
    """Test experiment configuration and variant generation."""
    print("Testing experiment configuration...")
    
    config = ULTRA_TINY_CONFIG
    print(f"Experiment: {config.experiment_name}")
    print(f"Parameter counts: {config.parameter_counts}")
    print(f"Bottleneck strengths: {config.bottleneck_strengths}")
    
    # Test model configs generation
    model_configs = config.get_model_configs()
    print(f"Generated {len(model_configs)} model configurations")
    for i, model_config in enumerate(model_configs):
        print(f"  Config {i}: {model_config.target_params} params -> estimated {model_config.estimated_params}")
    
    # Test variant generation
    variants = config.get_experiment_variants()
    print(f"Generated {len(variants)} experimental variants")
    
    # Validate configuration
    issues = config.validate_config()
    if issues:
        print("Configuration issues:", issues)
    else:
        print("Configuration validation passed")
    
    print("✓ Experiment configuration test passed\n")
    return True


def test_metrics_aggregator():
    """Test metrics aggregation functionality."""
    print("Testing metrics aggregator...")
    
    aggregator = MetricsAggregator()
    
    # Create mock experiment data
    for exp_id in ["exp1", "exp2"]:
        batches = []
        for step in range(5):
            # Create mock message batch
            messages = [[random.randint(0, 99) for _ in range(random.randint(2, 6))] for _ in range(20)]
            contexts = [{"step": step, "id": i} for i in range(20)]
            rewards = [random.random() for _ in range(20)]
            
            batch = MessageBatch(messages=messages, contexts=contexts, rewards=rewards)
            batches.append(batch)
        
        config = {
            "param_count": 1000000 if exp_id == "exp1" else 2000000,
            "bottleneck_strength": 0.3,
            "max_message_length": 8
        }
        
        # Add to aggregator
        result = aggregator.add_experiment_results(exp_id, batches, config)
        print(f"Added experiment {exp_id}: {len(result['batch_results'])} batches")
    
    # Test comparison
    comparison = aggregator.compare_experiments(["exp1", "exp2"])
    print("Comparison keys:", list(comparison.keys()))
    
    # Generate report
    report = aggregator.generate_metrics_report()
    print("Report keys:", list(report.keys()))
    print("Total experiments:", report["total_experiments"])
    
    print("✓ Metrics aggregator test passed\n")
    return True


def test_mini_experiment():
    """Run a very small experiment to test the full pipeline."""
    print("Testing mini experiment...")
    
    # Create minimal config
    config = ExperimentConfig(
        experiment_name="mini_test",
        parameter_counts=[500_000],  # Just one size
        bottleneck_strengths=[0.5],  # Just one bottleneck
        model_config=ModelConfig(
            d_model=16,
            n_heads=1,
            n_layers=1,
            vocab_size=100,
            max_seq_len=32,
            target_params=500_000
        ),
        training_config=TrainingConfig(
            batch_size=8,
            max_epochs=10,  # Very short training
            patience=5
        )
    )
    
    config.vocab_analysis_intervals = [5, 10]  # Analyze at these steps
    
    try:
        experiment = BottleneckExperiment(config, run_name="mini_test")
        
        # Get single variant
        variants = config.get_experiment_variants()
        variant = variants[0]
        
        print(f"Running mini experiment with {variant['param_count']} parameters...")
        
        # Run just one variant (not full experiment to save time)
        result = experiment.run_single_experiment(variant)
        
        print("Mini experiment completed successfully!")
        print(f"Final success rate: {result['analysis_report']['final_metrics']['success_rate']:.3f}")
        print(f"Training epochs: {len(result['training_history'])}")
        
    except Exception as e:
        print(f"Mini experiment failed: {e}")
        return False
    
    print("✓ Mini experiment test passed\n")
    return True


def run_all_tests():
    """Run all framework tests."""
    print("=" * 60)
    print("ULTRA-CONSTRAINED VOCABULARY EMERGENCE FRAMEWORK TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Environment & Metrics", test_environment_and_metrics),
        ("Vocabulary Analysis", test_vocabulary_analysis),
        ("Experiment Configuration", test_experiment_config),
        ("Metrics Aggregator", test_metrics_aggregator),
        ("Mini Experiment", test_mini_experiment)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            results.append((test_name, False))
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The framework is ready for experiments.")
        print("\nTo run a full experiment, use:")
        print("  python bottleneck_experiment.py")
        print("\nOr import and customize configurations:")
        print("  from experiment_config import ULTRA_TINY_CONFIG, SCALING_STUDY_CONFIG")
        print("  from bottleneck_experiment import BottleneckExperiment")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before running experiments.")
    
    return passed == total


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    success = run_all_tests()
    exit(0 if success else 1)