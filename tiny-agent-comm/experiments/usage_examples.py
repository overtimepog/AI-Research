# Usage Examples for Ultra-Constrained Vocabulary Framework

# Ultra Tiny Experiment
"""ultra_tiny_experiment"""

# Ultra-Tiny Model Experiment
from experiment_config import ULTRA_TINY_CONFIG
from bottleneck_experiment import BottleneckExperiment

experiment = BottleneckExperiment(ULTRA_TINY_CONFIG)
results = experiment.run_full_experiment()
print(f"Completed {len(results['individual_results'])} variants")


# Custom Experiment
"""custom_experiment"""

# Custom Experiment Configuration
from experiment_config import ExperimentConfig, ModelConfig

config = ExperimentConfig(
    experiment_name="my_experiment",
    parameter_counts=[1_000_000, 5_000_000],
    bottleneck_strengths=[0.2, 0.5],
    model_config=ModelConfig(d_model=32, n_layers=2)
)

experiment = BottleneckExperiment(config)
results = experiment.run_full_experiment()


# Vocabulary Analysis
"""vocabulary_analysis"""

# Vocabulary Analysis
from vocabulary_analysis import VocabularyAnalyzer, VocabularySnapshot

analyzer = VocabularyAnalyzer(vocab_size=1000, max_message_length=8)

# Add snapshots during training
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
print("Key findings:", report["key_findings"])


# Metrics Analysis
"""metrics_analysis"""

# Communication Metrics
from metrics import MessageBatch, CommunicationMetrics, BottleneckMetrics

batch = MessageBatch(
    messages=[[1, 2], [3, 4], [1, 3]],
    contexts=[{"id": i} for i in range(3)],
    rewards=[0.7, 0.8, 0.6]
)

comm_metrics = CommunicationMetrics()
basic = comm_metrics.compute_basic_metrics(batch)
entropy = comm_metrics.compute_entropy_metrics(batch)
efficiency = comm_metrics.compute_efficiency_metrics(batch)

print(f"Success rate: {basic['success_rate']:.2f}")
print(f"Token entropy: {entropy['token_entropy']:.2f}")
print(f"Length efficiency: {efficiency['length_efficiency']:.2f}")


