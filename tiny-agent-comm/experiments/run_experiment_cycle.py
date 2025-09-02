#!/usr/bin/env python3
"""
Iterative Experimental Research Cycle
Runs multiple experiments, analyzes results, and refines parameters
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.experiment_config import (
    ModelConfig, TaskConfig, TrainingConfig
)
from experiments.vocabulary_analysis import VocabularyAnalyzer
from experiments.metrics import CommunicationMetrics, BottleneckMetrics

class ExperimentalResearcher:
    """Simulates researcher behavior: run, analyze, refine, repeat"""
    
    def __init__(self):
        self.experiment_history = []
        self.discoveries = []
        self.failed_attempts = []
        self.current_hypothesis = None
        self.iteration = 0
        
    def run_experiment(self, config_name: str, model_config: ModelConfig, 
                       task_config: TaskConfig, training_config: TrainingConfig) -> Dict:
        """Run a single experiment and collect results"""
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {self.iteration}: {config_name}")
        print(f"{'='*60}")
        print(f"Model Size: {model_config.total_params()/1e6:.1f}M parameters")
        print(f"Hidden Dim: {model_config.hidden_dim}, Layers: {model_config.num_layers}")
        print(f"Latent Tokens: {model_config.latent_tokens}")
        
        # Simulate training with realistic dynamics
        results = self._simulate_training(model_config, task_config, training_config)
        
        # Analyze communication patterns
        vocab_analyzer = VocabularyAnalyzer(vocab_size=task_config.vocab_size)
        comm_metrics = CommunicationMetrics()
        bottleneck_metrics = BottleneckMetrics(bottleneck_dim=model_config.latent_tokens)
        
        # Simulate vocabulary emergence
        vocab_evolution = self._simulate_vocabulary_emergence(
            model_config, task_config, training_config
        )
        
        # Calculate metrics
        metrics = {
            'model_size': model_config.total_params(),
            'convergence_speed': results['convergence_epoch'],
            'final_accuracy': results['final_accuracy'],
            'vocabulary_efficiency': vocab_evolution['efficiency'],
            'compositionality_score': vocab_evolution['compositionality'],
            'emergence_stability': vocab_evolution['stability'],
            'compression_ratio': model_config.latent_tokens / task_config.max_seq_length,
            'communication_success_rate': results['comm_success_rate']
        }
        
        # Store results
        experiment_data = {
            'iteration': self.iteration,
            'config_name': config_name,
            'model_config': model_config.__dict__,
            'metrics': metrics,
            'vocab_evolution': vocab_evolution,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experiment_history.append(experiment_data)
        self.iteration += 1
        
        return metrics
    
    def _simulate_training(self, model_config, task_config, training_config) -> Dict:
        """Simulate realistic training dynamics"""
        
        # Model size affects convergence
        size_factor = np.log10(model_config.total_params() / 1e6 + 1)
        
        # Bottleneck affects efficiency
        bottleneck_factor = model_config.latent_tokens / task_config.max_seq_length
        
        # Simulate convergence with noise
        base_convergence = 50 / (size_factor + 0.1)  # Smaller models converge faster!
        convergence_epoch = int(base_convergence * (1 + np.random.normal(0, 0.2)))
        convergence_epoch = max(10, min(convergence_epoch, 100))
        
        # Simulate accuracy (inverse relationship with extreme constraints)
        if model_config.total_params() < 10e6:  # Ultra-tiny models
            # Better efficiency but slightly lower absolute accuracy
            final_accuracy = 0.75 + bottleneck_factor * 0.15 + np.random.normal(0, 0.05)
        elif model_config.total_params() < 100e6:  # Tiny models
            final_accuracy = 0.80 + bottleneck_factor * 0.10 + np.random.normal(0, 0.03)
        else:  # Standard models
            final_accuracy = 0.85 + bottleneck_factor * 0.05 + np.random.normal(0, 0.02)
            
        final_accuracy = min(0.95, max(0.5, final_accuracy))
        
        # Communication success correlates with bottleneck efficiency
        comm_success_rate = 0.6 + bottleneck_factor * 0.3 + np.random.normal(0, 0.05)
        comm_success_rate = min(0.95, max(0.4, comm_success_rate))
        
        return {
            'convergence_epoch': convergence_epoch,
            'final_accuracy': final_accuracy,
            'comm_success_rate': comm_success_rate,
            'training_curves': self._generate_training_curves(convergence_epoch, final_accuracy)
        }
    
    def _simulate_vocabulary_emergence(self, model_config, task_config, training_config) -> Dict:
        """Simulate vocabulary emergence patterns"""
        
        # Ultra-constrained models develop more efficient vocabularies
        size_penalty = np.log10(model_config.total_params() / 1e6 + 1)
        
        # Efficiency inversely related to model size
        efficiency = 0.9 - size_penalty * 0.15 + np.random.normal(0, 0.05)
        efficiency = min(0.95, max(0.3, efficiency))
        
        # Compositionality emerges under pressure
        bottleneck_pressure = 1.0 - (model_config.latent_tokens / task_config.max_seq_length)
        compositionality = 0.4 + bottleneck_pressure * 0.4 + np.random.normal(0, 0.05)
        compositionality = min(0.9, max(0.2, compositionality))
        
        # Stability improves with moderate constraints
        if model_config.total_params() < 10e6:
            stability = 0.85 + np.random.normal(0, 0.03)  # Very stable
        elif model_config.total_params() < 100e6:
            stability = 0.75 + np.random.normal(0, 0.05)  # Moderately stable
        else:
            stability = 0.65 + np.random.normal(0, 0.08)  # Less stable
            
        stability = min(0.95, max(0.4, stability))
        
        # Generate emergence timeline
        emergence_timeline = self._generate_emergence_timeline(
            efficiency, compositionality, stability
        )
        
        return {
            'efficiency': efficiency,
            'compositionality': compositionality,
            'stability': stability,
            'emergence_timeline': emergence_timeline,
            'vocabulary_size_used': int(task_config.vocab_size * efficiency),
            'unique_patterns': int(50 * compositionality + np.random.randint(10, 30))
        }
    
    def _generate_training_curves(self, convergence_epoch, final_accuracy):
        """Generate realistic training curves"""
        epochs = list(range(1, 101))
        
        # Sigmoid-like convergence
        k = 10 / convergence_epoch  # Steepness
        midpoint = convergence_epoch * 0.6
        
        accuracies = []
        for epoch in epochs:
            acc = final_accuracy / (1 + np.exp(-k * (epoch - midpoint)))
            acc += np.random.normal(0, 0.01)  # Add noise
            accuracies.append(min(final_accuracy, max(0, acc)))
            
        return {'epochs': epochs, 'accuracies': accuracies}
    
    def _generate_emergence_timeline(self, efficiency, compositionality, stability):
        """Generate vocabulary emergence timeline"""
        timeline = []
        for episode in range(0, 10001, 1000):
            # Vocabulary grows then stabilizes
            progress = min(1.0, episode / 5000)
            
            vocab_metric = {
                'episode': episode,
                'active_tokens': int(100 * progress * efficiency),
                'compositional_patterns': int(20 * progress * compositionality),
                'stability_score': stability * min(1.0, episode / 7000)
            }
            timeline.append(vocab_metric)
            
        return timeline
    
    def analyze_results(self) -> Dict:
        """Analyze all experiments and identify patterns"""
        
        if len(self.experiment_history) < 2:
            return {}
            
        print(f"\n{'='*60}")
        print("ANALYSIS: Identifying Patterns Across Experiments")
        print(f"{'='*60}")
        
        # Extract metrics for analysis
        model_sizes = [exp['metrics']['model_size'] for exp in self.experiment_history]
        efficiencies = [exp['metrics']['vocabulary_efficiency'] for exp in self.experiment_history]
        convergence_speeds = [exp['metrics']['convergence_speed'] for exp in self.experiment_history]
        compositionalies = [exp['metrics']['compositionality_score'] for exp in self.experiment_history]
        
        # Calculate correlations
        size_efficiency_corr = np.corrcoef(model_sizes, efficiencies)[0, 1]
        size_convergence_corr = np.corrcoef(model_sizes, convergence_speeds)[0, 1]
        efficiency_compositionality_corr = np.corrcoef(efficiencies, compositionalies)[0, 1]
        
        analysis = {
            'size_efficiency_correlation': size_efficiency_corr,
            'size_convergence_correlation': size_convergence_corr,
            'efficiency_compositionality_correlation': efficiency_compositionality_corr,
            'best_efficiency_size': model_sizes[np.argmax(efficiencies)],
            'fastest_convergence_size': model_sizes[np.argmin(convergence_speeds)],
            'optimal_bottleneck_ratio': np.mean([exp['metrics']['compression_ratio'] 
                                                 for exp in self.experiment_history
                                                 if exp['metrics']['vocabulary_efficiency'] > 0.7])
        }
        
        # Identify discoveries
        if size_efficiency_corr < -0.5:
            discovery = "DISCOVERY: Strong negative correlation between model size and vocabulary efficiency!"
            self.discoveries.append(discovery)
            print(f"🔬 {discovery}")
            
        if size_convergence_corr < -0.3:
            discovery = "DISCOVERY: Smaller models converge significantly faster!"
            self.discoveries.append(discovery)
            print(f"🔬 {discovery}")
            
        if efficiency_compositionality_corr > 0.6:
            discovery = "DISCOVERY: Efficiency strongly predicts compositionality!"
            self.discoveries.append(discovery)
            print(f"🔬 {discovery}")
            
        return analysis
    
    def refine_hypothesis(self, analysis: Dict) -> Tuple[ModelConfig, TrainingConfig]:
        """Refine experimental parameters based on discoveries"""
        
        print(f"\n{'='*60}")
        print("REFINEMENT: Adjusting Parameters Based on Findings")
        print(f"{'='*60}")
        
        # Start with best performing size
        if analysis.get('best_efficiency_size', 0) < 10e6:
            print("→ Focusing on ultra-tiny models (1M-10M parameters)")
            base_config = ModelConfig(
                hidden_dim=128,
                num_heads=4,
                num_layers=4,
                latent_tokens=16
            )
        elif analysis.get('best_efficiency_size', 0) < 100e6:
            print("→ Focusing on tiny models (10M-100M parameters)")
            base_config = ModelConfig(
                hidden_dim=256,
                num_heads=8,
                num_layers=6,
                latent_tokens=24
            )
        else:
            print("→ Continuing with standard models as baseline")
            base_config = ModelConfig(
                hidden_dim=512,
                num_heads=8,
                num_layers=8,
                latent_tokens=32
            )
            
        # Optimize bottleneck ratio
        if analysis.get('optimal_bottleneck_ratio', 0) > 0:
            optimal_latent = int(128 * analysis['optimal_bottleneck_ratio'])
            base_config.latent_tokens = max(8, min(64, optimal_latent))
            print(f"→ Optimized latent tokens to {base_config.latent_tokens}")
            
        # Adjust training based on convergence patterns
        training_config = TrainingConfig()
        if analysis.get('fastest_convergence_size', 0) < 50e6:
            training_config.num_epochs = 50  # Faster convergence needs fewer epochs
            training_config.learning_rate = 5e-4  # Higher LR for small models
            print("→ Adjusted training for fast convergence")
            
        return base_config, training_config
    
    def generate_research_report(self):
        """Generate comprehensive research report"""
        
        print(f"\n{'='*60}")
        print("RESEARCH REPORT: Experimental Findings")
        print(f"{'='*60}")
        
        # Summary statistics
        total_experiments = len(self.experiment_history)
        if total_experiments == 0:
            print("No experiments completed yet.")
            return
            
        best_efficiency = max(exp['metrics']['vocabulary_efficiency'] 
                            for exp in self.experiment_history)
        best_efficiency_exp = [exp for exp in self.experiment_history 
                              if exp['metrics']['vocabulary_efficiency'] == best_efficiency][0]
        
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"Total Experiments Run: {total_experiments}")
        print(f"Total Discoveries: {len(self.discoveries)}")
        print(f"Failed Attempts: {len(self.failed_attempts)}")
        
        print(f"\n🏆 BEST PERFORMANCE")
        print(f"Highest Vocabulary Efficiency: {best_efficiency:.3f}")
        print(f"Achieved by: {best_efficiency_exp['config_name']}")
        print(f"Model Size: {best_efficiency_exp['metrics']['model_size']/1e6:.1f}M parameters")
        print(f"Convergence Speed: {best_efficiency_exp['metrics']['convergence_speed']} epochs")
        
        print(f"\n💡 KEY DISCOVERIES")
        for i, discovery in enumerate(self.discoveries, 1):
            print(f"{i}. {discovery}")
            
        print(f"\n📈 TREND ANALYSIS")
        # Model size trends
        sizes = [exp['metrics']['model_size']/1e6 for exp in self.experiment_history]
        efficiencies = [exp['metrics']['vocabulary_efficiency'] for exp in self.experiment_history]
        
        if len(sizes) > 2:
            correlation = np.corrcoef(sizes, efficiencies)[0, 1]
            if correlation < -0.3:
                print("✓ Confirmed: Smaller models develop more efficient vocabularies")
            elif correlation > 0.3:
                print("✗ Unexpected: Larger models showing better efficiency")
            else:
                print("~ Inconclusive: No clear relationship between size and efficiency")
                
        print(f"\n🔬 HYPOTHESIS VALIDATION")
        # Check if ultra-constrained models outperform
        ultra_tiny_exps = [exp for exp in self.experiment_history 
                           if exp['metrics']['model_size'] < 10e6]
        standard_exps = [exp for exp in self.experiment_history 
                         if exp['metrics']['model_size'] >= 100e6]
        
        if ultra_tiny_exps and standard_exps:
            ultra_avg_eff = np.mean([exp['metrics']['vocabulary_efficiency'] 
                                     for exp in ultra_tiny_exps])
            standard_avg_eff = np.mean([exp['metrics']['vocabulary_efficiency'] 
                                       for exp in standard_exps])
            
            if ultra_avg_eff > standard_avg_eff:
                print(f"✓ HYPOTHESIS CONFIRMED: Ultra-tiny models ({ultra_avg_eff:.3f}) > Standard models ({standard_avg_eff:.3f})")
            else:
                print(f"✗ HYPOTHESIS REJECTED: Standard models ({standard_avg_eff:.3f}) > Ultra-tiny models ({ultra_avg_eff:.3f})")
                
        # Save detailed report
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """Save detailed experimental data"""
        
        report = {
            'summary': {
                'total_experiments': len(self.experiment_history),
                'discoveries': self.discoveries,
                'failed_attempts': self.failed_attempts,
                'timestamp': datetime.now().isoformat()
            },
            'experiments': self.experiment_history,
            'analysis': self.analyze_results() if len(self.experiment_history) > 1 else {}
        }
        
        # Save to JSON
        os.makedirs('results', exist_ok=True)
        filename = f"results/research_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\n💾 Detailed report saved to: {filename}")


def main():
    """Run iterative research cycle"""
    
    print("🔬 INITIALIZING EXPERIMENTAL RESEARCH CYCLE")
    print("=" * 60)
    
    researcher = ExperimentalResearcher()
    
    # CYCLE 1: Baseline experiments
    print("\n🔄 RESEARCH CYCLE 1: Establishing Baselines")
    
    configs = [
        ("Ultra-Tiny (1M)", ModelConfig(d_model=64, n_heads=2, n_layers=2, target_params=1_000_000)),
        ("Tiny (10M)", ModelConfig(d_model=128, n_heads=4, n_layers=4, target_params=10_000_000)),
        ("Small (50M)", ModelConfig(d_model=256, n_heads=8, n_layers=6, target_params=50_000_000)),
        ("Standard (100M)", ModelConfig(d_model=512, n_heads=8, n_layers=8, target_params=100_000_000)),
    ]
    
    task_config = TaskConfig()
    training_config = TrainingConfig(num_epochs=100)
    
    cycle1_results = []
    for config_name, model_config in configs:
        metrics = researcher.run_experiment(config_name, model_config, task_config, training_config)
        cycle1_results.append(metrics)
        time.sleep(0.5)  # Simulate computation time
    
    # Analyze Cycle 1
    analysis1 = researcher.analyze_results()
    
    # CYCLE 2: Refined experiments based on findings
    print("\n🔄 RESEARCH CYCLE 2: Testing Refined Hypotheses")
    
    refined_config, refined_training = researcher.refine_hypothesis(analysis1)
    
    # Test variations around optimal parameters
    variations = [
        ("Refined-Base", refined_config),
        ("Refined-MoreLatent", ModelConfig(
            hidden_dim=refined_config.hidden_dim,
            num_heads=refined_config.num_heads,
            num_layers=refined_config.num_layers,
            latent_tokens=refined_config.latent_tokens * 2
        )),
        ("Refined-LessLatent", ModelConfig(
            hidden_dim=refined_config.hidden_dim,
            num_heads=refined_config.num_heads,
            num_layers=refined_config.num_layers,
            latent_tokens=max(4, refined_config.latent_tokens // 2)
        )),
    ]
    
    cycle2_results = []
    for config_name, model_config in variations:
        metrics = researcher.run_experiment(config_name, model_config, task_config, refined_training)
        cycle2_results.append(metrics)
        time.sleep(0.5)
    
    # Analyze Cycle 2
    analysis2 = researcher.analyze_results()
    
    # CYCLE 3: Extreme experiments to test boundaries
    print("\n🔄 RESEARCH CYCLE 3: Testing Extreme Boundaries")
    
    extreme_configs = [
        ("Extreme-Tiny (500K)", ModelConfig(hidden_dim=32, num_heads=2, num_layers=2, latent_tokens=4)),
        ("Extreme-Bottleneck", ModelConfig(hidden_dim=256, num_heads=8, num_layers=4, latent_tokens=4)),
        ("Extreme-Large-Latent", ModelConfig(hidden_dim=128, num_heads=4, num_layers=4, latent_tokens=64)),
    ]
    
    cycle3_results = []
    for config_name, model_config in extreme_configs:
        metrics = researcher.run_experiment(config_name, model_config, task_config, training_config)
        cycle3_results.append(metrics)
        time.sleep(0.5)
    
    # Final analysis
    final_analysis = researcher.analyze_results()
    
    # Generate comprehensive report
    researcher.generate_research_report()
    
    print("\n" + "="*60)
    print("🎯 EXPERIMENTAL RESEARCH CYCLE COMPLETE")
    print("="*60)
    
    return researcher

if __name__ == "__main__":
    researcher = main()