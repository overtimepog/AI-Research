# Ultra-Constrained Model Vocabulary Emergence Framework

This directory contains a comprehensive experimental validation framework for testing the hypothesis that **ultra-constrained models (1M-10M parameters) develop more efficient emergent vocabularies under information bottleneck constraints**.

## Framework Overview

The framework is designed to systematically test communication protocol emergence across different model scales and bottleneck strengths, providing comprehensive analysis of vocabulary structure, compositionality, and efficiency.

### Core Hypothesis
**Smaller models with tighter communication constraints develop more compositional and efficient vocabularies faster than larger models.**

## Framework Components

### 1. `experiment_config.py` - Configuration System
- **ModelConfig**: Ultra-tiny transformer architectures (1M-500M parameters)
- **TaskConfig**: Referential communication game setups
- **TrainingConfig**: Training dynamics with curriculum learning
- **ExperimentConfig**: Master configuration with parameter sweeps

**Key Features:**
- Automatic parameter scaling to hit target model sizes
- Bottleneck strength sweeps (0.1 to 1.0)
- Predefined configurations: `ULTRA_TINY_CONFIG`, `SCALING_STUDY_CONFIG`, `BOTTLENECK_STUDY_CONFIG`

### 2. `vocabulary_analysis.py` - Vocabulary Analysis Tools
- **VocabularySnapshot**: Captures vocabulary state at training checkpoints
- **VocabularyAnalyzer**: Comprehensive analysis of emergent structures

**Analysis Capabilities:**
- Entropy metrics (token, message, positional)
- Compositionality measures (systematicity, productivity)
- Efficiency metrics (length efficiency, bit efficiency)
- Emergence dynamics tracking
- Parameter scaling analysis

### 3. `bottleneck_experiment.py` - Main Experiment Runner
- **UltraTinyTransformer**: Minimal transformer implementation
- **ReferentialGameEnvironment**: Communication task environment
- **BottleneckExperiment**: Complete experimental pipeline

**Experiment Features:**
- Systematic parameter count sweeps
- Bottleneck strength variations  
- Curriculum learning with temperature scheduling
- Comprehensive result aggregation and analysis

### 4. `metrics.py` - Communication Metrics Suite
- **CommunicationMetrics**: Basic communication analysis
- **BottleneckMetrics**: Information bottleneck specific metrics
- **MetricsAggregator**: Cross-experiment analysis

**Metric Categories:**
- Basic: success rate, message diversity, vocabulary usage
- Entropy: token/message entropy, perplexity, compression ratios
- Efficiency: length/bit efficiency, Pareto efficiency
- Compositionality: systematicity, productivity, generalization
- Convergence: protocol stability, consensus measures

## Quick Start

### Installation

```bash
# Install required dependencies
pip install torch>=2.0.0 numpy pandas scikit-learn matplotlib seaborn scipy networkx

# Validate installation
python validate_imports.py
```

### Running Experiments

#### 1. Ultra-Tiny Model Study
```python
from experiment_config import ULTRA_TINY_CONFIG
from bottleneck_experiment import BottleneckExperiment

# Run ultra-constrained models (500K-2M parameters)
experiment = BottleneckExperiment(ULTRA_TINY_CONFIG)
results = experiment.run_full_experiment()
```

#### 2. Parameter Scaling Study
```python
from experiment_config import SCALING_STUDY_CONFIG
from bottleneck_experiment import BottleneckExperiment

# Test scaling from 1M to 500M parameters
experiment = BottleneckExperiment(SCALING_STUDY_CONFIG)
results = experiment.run_full_experiment()
```

#### 3. Bottleneck Strength Study
```python
from experiment_config import BOTTLENECK_STUDY_CONFIG
from bottleneck_experiment import BottleneckExperiment

# Test different bottleneck strengths on fixed model size
experiment = BottleneckExperiment(BOTTLENECK_STUDY_CONFIG)
results = experiment.run_full_experiment()
```

### Custom Experiments

```python
from experiment_config import ExperimentConfig, ModelConfig

# Create custom configuration
config = ExperimentConfig(
    experiment_name="custom_study",
    parameter_counts=[1_000_000, 5_000_000],
    bottleneck_strengths=[0.2, 0.5],
    model_config=ModelConfig(
        d_model=32,
        n_layers=2,
        vocab_size=500,
        max_seq_len=64
    )
)

experiment = BottleneckExperiment(config)
results = experiment.run_full_experiment()
```

## Framework Architecture

### Model Architecture: UltraTinyTransformer
- Minimal parameter transformer with automatic scaling
- Attention heads: 1-4, Layers: 1-3, d_model: 16-128
- Vocabulary: 100-1000 tokens
- Sequence length: 32-128 tokens

### Communication Task: Referential Game
- Grid world with objects having multiple properties
- Speaker generates message to identify target object
- Success measured by listener's identification accuracy
- Bottleneck constraints on message length and vocabulary

### Training Protocol
- Curriculum learning with increasing task difficulty
- Temperature scheduling for exploration/exploitation
- Regularization for compression and vocabulary diversity
- Early stopping based on convergence criteria

## Analysis Pipeline

### 1. Real-time Monitoring
- Vocabulary usage tracking
- Success rate evolution
- Message length optimization
- Protocol convergence detection

### 2. Compositional Analysis
- Systematic generalization tests
- Constituent swapping analysis  
- Hierarchical structure detection
- Productivity measurements

### 3. Efficiency Analysis
- Compression ratio computation
- Information density measures
- Pareto efficiency frontiers
- Rate-distortion analysis

### 4. Scaling Analysis
- Parameter count vs. performance curves
- Bottleneck strength effects
- Convergence speed comparisons
- Statistical significance testing

## Expected Results

Based on the core hypothesis, the framework should reveal:

### Ultra-Constrained Models (1M-5M parameters)
- **Faster convergence** to stable protocols
- **Higher compression ratios** (more efficient vocabularies)
- **Better compositionality** under tight bottlenecks
- **More systematic generalization**

### Scaling Effects
- **Inverse relationship** between model size and compression efficiency
- **Optimal bottleneck strength** around 0.2-0.4 for small models
- **Diminishing returns** for models >50M parameters in constrained settings

### Bottleneck Effects
- **Tighter bottlenecks** → better compositional structure
- **Communication efficiency** peaks at intermediate bottleneck strengths
- **Vocabulary utilization** becomes more focused under constraints

## Output and Visualization

The framework generates comprehensive outputs:

### Result Files
- `complete_results.json`: Full experimental data
- `vocabulary_analyzers.pkl`: Detailed vocabulary analysis objects
- `result_{param_count}_{bottleneck}.json`: Individual variant results

### Visualizations
- Vocabulary evolution plots
- Parameter scaling curves
- Bottleneck effect heatmaps
- Compositionality development graphs

### Statistical Analysis
- Correlation analyses between parameters and outcomes
- Statistical significance tests
- Effect size computations
- Confidence intervals

## Validation and Testing

### Framework Validation
```bash
# Run comprehensive framework tests
python test_framework.py

# Quick validation of imports and basic functionality
python validate_imports.py
```

### Test Coverage
- Model creation and parameter counting
- Environment and task generation
- Metrics computation accuracy
- Analysis pipeline functionality
- Configuration validation
- Results aggregation

## Extensibility

The framework is designed for easy extension:

### Adding New Tasks
```python
class NewTaskEnvironment:
    def sample_task(self): # Generate task instances
    def evaluate_message(self, message, context, target): # Evaluate success
```

### Adding New Metrics
```python
def compute_custom_metric(self, batch: MessageBatch) -> Dict[str, float]:
    # Custom analysis logic
    return {"custom_score": score}
```

### Adding New Models
```python
class CustomModel(nn.Module):
    def generate(self, context, max_length, temperature): # Generate messages
```

## Performance Optimization

For large-scale experiments:
- Use GPU acceleration when available
- Implement distributed training for multiple variants
- Cache vocabulary analysis computations
- Use streaming metrics for memory efficiency

## Citation

If you use this framework, please cite:

```bibtex
@misc{ultra_constrained_vocab_2025,
  title={Ultra-Constrained Model Vocabulary Emergence Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/tiny-agent-comm}
}
```

## License

MIT License - see LICENSE file for details.

---

This framework provides a comprehensive platform for investigating how resource constraints shape emergent communication protocols, with particular focus on the efficiency advantages of ultra-constrained models.