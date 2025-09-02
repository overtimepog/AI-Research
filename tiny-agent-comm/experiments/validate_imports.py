"""
Simple validation script to check if all modules can be imported successfully.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    import_tests = [
        ("torch", "import torch"),
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("sklearn", "from sklearn.metrics import mutual_info_score"),
        ("networkx", "import networkx as nx"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("seaborn", "import seaborn as sns"),
        ("scipy", "from scipy import stats"),
        ("experiment_config", "from experiment_config import ExperimentConfig, ModelConfig"),
        ("vocabulary_analysis", "from vocabulary_analysis import VocabularyAnalyzer"),
        ("metrics", "from metrics import CommunicationMetrics, BottleneckMetrics"),
    ]
    
    results = []
    for name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            results.append((name, True, None))
            print(f"✓ {name}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name}: {str(e)}")
    
    return results

def test_basic_functionality():
    """Test basic functionality of our modules."""
    try:
        from experiment_config import ModelConfig, ULTRA_TINY_CONFIG
        
        # Test model config
        config = ModelConfig(target_params=1000000)
        scaled = config.scale_to_target()
        print(f"✓ Model config scaling: {scaled.estimated_params} params")
        
        # Test experiment config
        variants = ULTRA_TINY_CONFIG.get_experiment_variants()
        print(f"✓ Experiment variants: {len(variants)} variants generated")
        
        # Test vocabulary analyzer
        from vocabulary_analysis import VocabularyAnalyzer, VocabularySnapshot
        analyzer = VocabularyAnalyzer(vocab_size=100, max_message_length=8)
        
        # Create dummy snapshot
        snapshot = VocabularySnapshot(
            step=100,
            messages=[[1, 2, 3], [4, 5], [1, 6]],
            contexts=[{"id": i} for i in range(3)],
            rewards=[0.8, 0.6, 0.9],
            model_params=1000000,
            bottleneck_strength=0.3
        )
        analyzer.add_snapshot(snapshot)
        report = analyzer.generate_analysis_report()
        print(f"✓ Vocabulary analysis: {len(report['key_findings'])} findings")
        
        # Test metrics
        from metrics import MessageBatch, CommunicationMetrics
        batch = MessageBatch(
            messages=[[1, 2], [3, 4], [1, 3]],
            contexts=[{"id": i} for i in range(3)],
            rewards=[0.7, 0.8, 0.6]
        )
        comm_metrics = CommunicationMetrics()
        basic_metrics = comm_metrics.compute_basic_metrics(batch)
        print(f"✓ Metrics computation: success_rate={basic_metrics['success_rate']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("FRAMEWORK IMPORT VALIDATION")
    print("=" * 50)
    
    # Test imports
    results = test_imports()
    
    failed_imports = [name for name, success, _ in results if not success]
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("Some functionality may not work properly.")
    else:
        print("\n🎉 All imports successful!")
    
    print("\n" + "=" * 50)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test basic functionality
    if test_basic_functionality():
        print("\n🎉 Basic functionality test passed!")
        print("\nThe framework appears to be working correctly.")
        print("You can now run experiments with:")
        print("  python3 bottleneck_experiment.py")
    else:
        print("\n⚠️  Basic functionality test failed.")
        print("Please check the error messages above.")
    
    print("\n" + "=" * 50)