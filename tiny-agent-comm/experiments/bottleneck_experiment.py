"""
Main experiment script for testing information bottleneck effects on vocabulary emergence.

This module implements the core experimental framework for testing the hypothesis
that ultra-constrained models develop more efficient emergent vocabularies under
information bottleneck constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
import wandb
from pathlib import Path
import json
import pickle
from dataclasses import asdict
from tqdm import tqdm
import logging
from collections import defaultdict
import random

from experiment_config import ExperimentConfig, ModelConfig, TaskConfig, TrainingConfig, VocabularySnapshot
from vocabulary_analysis import VocabularyAnalyzer
from metrics import CommunicationMetrics, BottleneckMetrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UltraTinyTransformer(nn.Module):
    """Ultra-constrained transformer model for communication experiments."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
        # Log actual parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model created with {total_params:,} parameters (target: {config.target_params:,})")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(pos_ids)
        
        hidden_states = self.dropout(token_emb + pos_emb)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        return logits
    
    def generate(self, context: torch.Tensor, max_length: int = 8, 
                temperature: float = 1.0, do_sample: bool = True) -> List[int]:
        """Generate a message given context."""
        self.eval()
        with torch.no_grad():
            # Start with a special start token (assume token 0)
            generated = [0]  # Start token
            
            for _ in range(max_length - 1):
                input_ids = torch.tensor([generated], device=context.device)
                logits = self.forward(input_ids)
                
                # Get next token logits and apply temperature
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = next_token_logits.argmax().item()
                
                generated.append(next_token)
                
                # Stop at end token (assume token 1)
                if next_token == 1:
                    break
            
            return generated[1:]  # Remove start token


class TransformerLayer(nn.Module):
    """Single transformer layer with ultra-tiny configuration."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        self.attn_norm = nn.LayerNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, attention_mask)
        x = self.attn_norm(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_output))
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.config.n_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.config.n_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.config.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores.masked_fill_(attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.output(attn_output)


class FeedForwardNetwork(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class ReferentialGameEnvironment:
    """Environment for referential communication games."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        # Generate objects with properties
        self.objects = []
        for i in range(self.config.n_objects):
            obj = {
                'id': i,
                'properties': [
                    random.randint(0, self.config.property_values - 1) 
                    for _ in range(self.config.n_properties)
                ]
            }
            self.objects.append(obj)
    
    def sample_task(self) -> Tuple[Dict, int]:
        """Sample a referential task: context and target object."""
        target_idx = random.randint(0, len(self.objects) - 1)
        target_obj = self.objects[target_idx]
        
        # Create context with distractors
        n_distractors = min(3, len(self.objects) - 1)
        distractor_indices = random.sample(
            [i for i in range(len(self.objects)) if i != target_idx],
            n_distractors
        )
        
        context = {
            'target': target_obj,
            'distractors': [self.objects[i] for i in distractor_indices],
            'target_idx': target_idx
        }
        
        return context, target_idx
    
    def evaluate_message(self, message: List[int], context: Dict, target_idx: int) -> float:
        """Evaluate how well a message identifies the target."""
        # Simplified evaluation: success if message is unique and consistent
        # In practice, this would involve a listener model
        
        # For now, give partial credit based on message properties
        score = 0.0
        
        # Length penalty for very short or very long messages
        msg_len = len(message)
        if 2 <= msg_len <= self.config.max_message_length:
            score += 0.5
        
        # Consistency bonus (simplified)
        if len(set(message)) == len(message):  # All unique tokens
            score += 0.3
        
        # Random component to simulate listener success
        score += random.random() * 0.2
        
        return min(1.0, score)


class CommunicationDataset(Dataset):
    """Dataset for communication tasks."""
    
    def __init__(self, environment: ReferentialGameEnvironment, n_samples: int = 1000):
        self.environment = environment
        self.samples = []
        
        # Pre-generate samples
        for _ in range(n_samples):
            context, target_idx = environment.sample_task()
            self.samples.append((context, target_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class BottleneckExperiment:
    """Main experiment class for testing information bottleneck effects."""
    
    def __init__(self, config: ExperimentConfig, run_name: Optional[str] = None):
        self.config = config
        self.run_name = run_name or f"bottleneck_experiment_{config.experiment_name}"
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # Initialize metrics tracking
        self.comm_metrics = CommunicationMetrics()
        self.bottleneck_metrics = BottleneckMetrics()
        
        # Results storage
        self.results = defaultdict(list)
        self.vocabulary_analyzers = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup experiment logging."""
        # Create experiment directory
        self.exp_dir = Path(f"results/{self.run_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
            wandb.init(
                project="ultra-constrained-vocab-emergence",
                name=self.run_name,
                config=asdict(self.config)
            )
    
    def run_single_experiment(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experimental variant."""
        logger.info(f"Running experiment variant: {variant['param_count']} params, "
                   f"bottleneck {variant['bottleneck_strength']}")
        
        # Extract configurations
        model_config = variant['model_config']
        task_config = variant['task_config']
        training_config = variant['training_config']
        
        # Initialize components
        model = UltraTinyTransformer(model_config).to(self.config.device)
        environment = ReferentialGameEnvironment(task_config)
        optimizer = Adam(model.parameters(), lr=training_config.learning_rate)
        
        # Initialize vocabulary analyzer
        vocab_analyzer = VocabularyAnalyzer(
            vocab_size=model_config.vocab_size,
            max_message_length=task_config.max_message_length
        )
        
        # Training loop
        model.train()
        epoch_results = []
        
        for epoch in tqdm(range(training_config.max_epochs), desc="Training"):
            # Generate batch of tasks
            batch_contexts = []
            batch_targets = []
            batch_messages = []
            batch_rewards = []
            
            for _ in range(training_config.batch_size):
                context, target_idx = environment.sample_task()
                
                # Generate message
                # Convert context to tensor (simplified)
                context_tensor = torch.zeros(1, model_config.d_model, device=self.config.device)
                message = model.generate(context_tensor, 
                                       max_length=task_config.max_message_length,
                                       temperature=self._get_temperature(epoch, training_config))
                
                # Evaluate message
                reward = environment.evaluate_message(message, context, target_idx)
                
                batch_contexts.append(context)
                batch_targets.append(target_idx)
                batch_messages.append(message)
                batch_rewards.append(reward)
            
            # Compute loss and update model
            loss = self._compute_communication_loss(
                model, batch_messages, batch_rewards, training_config
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            epoch_result = {
                'epoch': epoch,
                'loss': loss.item(),
                'avg_reward': np.mean(batch_rewards),
                'avg_message_length': np.mean([len(msg) for msg in batch_messages]),
                'unique_messages': len(set(tuple(msg) for msg in batch_messages))
            }
            epoch_results.append(epoch_result)
            
            # Vocabulary analysis at specified intervals
            if epoch in self.config.vocab_analysis_intervals:
                snapshot = VocabularySnapshot(
                    step=epoch,
                    messages=batch_messages,
                    contexts=batch_contexts,
                    rewards=batch_rewards,
                    model_params=variant['param_count'],
                    bottleneck_strength=variant['bottleneck_strength']
                )
                vocab_analyzer.add_snapshot(snapshot)
            
            # Early stopping check
            if epoch > training_config.patience:
                recent_rewards = [r['avg_reward'] for r in epoch_results[-training_config.patience:]]
                if max(recent_rewards) - min(recent_rewards) < 0.01:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final analysis
        final_snapshot = VocabularySnapshot(
            step=training_config.max_epochs,
            messages=batch_messages,
            contexts=batch_contexts, 
            rewards=batch_rewards,
            model_params=variant['param_count'],
            bottleneck_strength=variant['bottleneck_strength']
        )
        vocab_analyzer.add_snapshot(final_snapshot)
        
        # Generate analysis report
        analysis_report = vocab_analyzer.generate_analysis_report()
        
        # Save results
        variant_key = f"{variant['param_count']}_{variant['bottleneck_strength']}"
        self.vocabulary_analyzers[variant_key] = vocab_analyzer
        
        result = {
            'variant': variant,
            'training_history': epoch_results,
            'analysis_report': analysis_report,
            'final_model_state': model.state_dict()
        }
        
        # Save individual result
        result_file = self.exp_dir / f"result_{variant_key}.json"
        with open(result_file, 'w') as f:
            # Remove non-serializable items for JSON
            json_result = {k: v for k, v in result.items() if k != 'final_model_state'}
            json.dump(json_result, f, indent=2, default=str)
        
        return result
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experimental suite."""
        logger.info("Starting full bottleneck experiment suite")
        
        # Get all experimental variants
        variants = self.config.get_experiment_variants()
        logger.info(f"Running {len(variants)} experimental variants")
        
        all_results = []
        
        for i, variant in enumerate(variants):
            logger.info(f"Running variant {i+1}/{len(variants)}")
            try:
                result = self.run_single_experiment(variant)
                all_results.append(result)
                
                # Log to wandb if enabled
                if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
                    wandb.log({
                        "variant_completed": i+1,
                        "param_count": variant['param_count'],
                        "bottleneck_strength": variant['bottleneck_strength'],
                        "final_success_rate": result['analysis_report']['final_metrics']['success_rate']
                    })
                    
            except Exception as e:
                logger.error(f"Error in variant {i+1}: {str(e)}")
                continue
        
        # Aggregate analysis
        aggregate_results = self.analyze_aggregate_results(all_results)
        
        # Save complete results
        complete_results = {
            'config': asdict(self.config),
            'individual_results': all_results,
            'aggregate_analysis': aggregate_results
        }
        
        results_file = self.exp_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Save vocabulary analyzers
        analyzers_file = self.exp_dir / "vocabulary_analyzers.pkl"
        with open(analyzers_file, 'wb') as f:
            pickle.dump(self.vocabulary_analyzers, f)
        
        logger.info(f"Experiment complete. Results saved to {self.exp_dir}")
        return complete_results
    
    def analyze_aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results across all experimental variants."""
        logger.info("Analyzing aggregate results")
        
        # Group results by parameter count and bottleneck strength
        by_params = defaultdict(list)
        by_bottleneck = defaultdict(list)
        
        for result in all_results:
            variant = result['variant']
            param_count = variant['param_count']
            bottleneck = variant['bottleneck_strength']
            
            by_params[param_count].append(result)
            by_bottleneck[bottleneck].append(result)
        
        # Parameter scaling analysis
        param_scaling = self.analyze_parameter_scaling(by_params)
        
        # Bottleneck effect analysis
        bottleneck_effects = self.analyze_bottleneck_effects(by_bottleneck)
        
        # Statistical tests
        statistical_tests = self.run_statistical_tests(all_results)
        
        return {
            'parameter_scaling': param_scaling,
            'bottleneck_effects': bottleneck_effects,
            'statistical_tests': statistical_tests,
            'summary_statistics': self.compute_summary_statistics(all_results)
        }
    
    def analyze_parameter_scaling(self, by_params: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Analyze how metrics scale with parameter count."""
        scaling_data = {
            'param_counts': [],
            'final_success_rates': [],
            'final_vocab_sizes': [],
            'final_compositionality': [],
            'convergence_speeds': []
        }
        
        for param_count, results in by_params.items():
            # Average across different bottleneck strengths
            success_rates = []
            vocab_sizes = []
            comp_scores = []
            conv_speeds = []
            
            for result in results:
                report = result['analysis_report']
                success_rates.append(report['final_metrics']['success_rate'])
                vocab_sizes.append(report['final_metrics'].get('vocab_utilization', 0) * 1000)
                comp_scores.append(report['final_metrics']['compositionality_score'])
                
                # Estimate convergence speed from training history
                history = result['training_history']
                rewards = [h['avg_reward'] for h in history]
                final_reward = rewards[-1] if rewards else 0
                target_reward = 0.8 * final_reward
                
                conv_speed = len(rewards)
                for i, reward in enumerate(rewards):
                    if reward >= target_reward:
                        conv_speed = i
                        break
                conv_speeds.append(conv_speed)
            
            scaling_data['param_counts'].append(param_count)
            scaling_data['final_success_rates'].append(np.mean(success_rates))
            scaling_data['final_vocab_sizes'].append(np.mean(vocab_sizes))
            scaling_data['final_compositionality'].append(np.mean(comp_scores))
            scaling_data['convergence_speeds'].append(np.mean(conv_speeds))
        
        return scaling_data
    
    def analyze_bottleneck_effects(self, by_bottleneck: Dict[float, List[Dict]]) -> Dict[str, Any]:
        """Analyze effects of different bottleneck strengths."""
        bottleneck_data = {
            'bottleneck_strengths': [],
            'compression_ratios': [],
            'efficiency_scores': [],
            'vocab_diversity': []
        }
        
        for bottleneck, results in by_bottleneck.items():
            compression_ratios = []
            efficiency_scores = []
            vocab_diversities = []
            
            for result in results:
                report = result['analysis_report']
                compression_ratios.append(report['final_metrics'].get('compression_ratio', 0))
                efficiency_scores.append(report['final_metrics']['length_efficiency'])
                vocab_diversities.append(report['final_metrics']['vocab_utilization'])
            
            bottleneck_data['bottleneck_strengths'].append(bottleneck)
            bottleneck_data['compression_ratios'].append(np.mean(compression_ratios))
            bottleneck_data['efficiency_scores'].append(np.mean(efficiency_scores))
            bottleneck_data['vocab_diversity'].append(np.mean(vocab_diversities))
        
        return bottleneck_data
    
    def run_statistical_tests(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run statistical tests on the results."""
        from scipy import stats
        
        # Extract key metrics
        param_counts = []
        bottleneck_strengths = []
        success_rates = []
        comp_scores = []
        
        for result in all_results:
            variant = result['variant']
            report = result['analysis_report']
            
            param_counts.append(np.log10(variant['param_count']))  # Log scale
            bottleneck_strengths.append(variant['bottleneck_strength'])
            success_rates.append(report['final_metrics']['success_rate'])
            comp_scores.append(report['final_metrics']['compositionality_score'])
        
        tests = {}
        
        # Correlation between parameters and performance
        if len(param_counts) > 2:
            param_success_corr, param_success_p = stats.pearsonr(param_counts, success_rates)
            param_comp_corr, param_comp_p = stats.pearsonr(param_counts, comp_scores)
            
            tests['parameter_correlations'] = {
                'param_success_correlation': param_success_corr,
                'param_success_p_value': param_success_p,
                'param_compositionality_correlation': param_comp_corr,
                'param_compositionality_p_value': param_comp_p
            }
        
        # Effect of bottleneck strength
        if len(bottleneck_strengths) > 2:
            bottleneck_success_corr, bottleneck_success_p = stats.pearsonr(bottleneck_strengths, success_rates)
            bottleneck_comp_corr, bottleneck_comp_p = stats.pearsonr(bottleneck_strengths, comp_scores)
            
            tests['bottleneck_correlations'] = {
                'bottleneck_success_correlation': bottleneck_success_corr,
                'bottleneck_success_p_value': bottleneck_success_p,
                'bottleneck_compositionality_correlation': bottleneck_comp_corr,
                'bottleneck_compositionality_p_value': bottleneck_comp_p
            }
        
        return tests
    
    def compute_summary_statistics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        success_rates = []
        comp_scores = []
        vocab_utilizations = []
        convergence_epochs = []
        
        for result in all_results:
            report = result['analysis_report']
            success_rates.append(report['final_metrics']['success_rate'])
            comp_scores.append(report['final_metrics']['compositionality_score'])
            vocab_utilizations.append(report['final_metrics']['vocab_utilization'])
            
            # Estimate convergence
            history = result['training_history']
            rewards = [h['avg_reward'] for h in history]
            final_reward = rewards[-1] if rewards else 0
            target_reward = 0.9 * final_reward
            
            convergence_epoch = len(rewards)
            for i, reward in enumerate(rewards):
                if reward >= target_reward:
                    convergence_epoch = i
                    break
            convergence_epochs.append(convergence_epoch)
        
        return {
            'success_rate_stats': {
                'mean': np.mean(success_rates),
                'std': np.std(success_rates),
                'min': np.min(success_rates),
                'max': np.max(success_rates)
            },
            'compositionality_stats': {
                'mean': np.mean(comp_scores),
                'std': np.std(comp_scores),
                'min': np.min(comp_scores),
                'max': np.max(comp_scores)
            },
            'vocab_utilization_stats': {
                'mean': np.mean(vocab_utilizations),
                'std': np.std(vocab_utilizations),
                'min': np.min(vocab_utilizations),
                'max': np.max(vocab_utilizations)
            },
            'convergence_stats': {
                'mean_epochs': np.mean(convergence_epochs),
                'std_epochs': np.std(convergence_epochs),
                'fastest_convergence': np.min(convergence_epochs),
                'slowest_convergence': np.max(convergence_epochs)
            },
            'total_experiments': len(all_results)
        }
    
    def _get_temperature(self, epoch: int, training_config: TrainingConfig) -> float:
        """Get temperature for current epoch based on schedule."""
        for epoch_threshold, temp in training_config.temperature_schedule:
            if epoch >= epoch_threshold:
                temperature = temp
        return temperature
    
    def _compute_communication_loss(self, model: UltraTinyTransformer, 
                                  messages: List[List[int]], rewards: List[float],
                                  training_config: TrainingConfig) -> torch.Tensor:
        """Compute communication loss with bottleneck regularization."""
        batch_size = len(messages)
        device = model.token_embedding.weight.device
        
        # Convert messages to tensor (pad to max length)
        max_len = max(len(msg) for msg in messages) if messages else 1
        padded_messages = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, msg in enumerate(messages):
            msg_len = min(len(msg), max_len)
            padded_messages[i, :msg_len] = torch.tensor(msg[:msg_len], device=device)
        
        # Forward pass
        logits = model(padded_messages)
        
        # Reconstruction loss (simplified - in practice would use listener model)
        targets = padded_messages.roll(-1, dims=1)  # Next token prediction
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        
        # Reward-weighted loss
        rewards_tensor = torch.tensor(rewards, device=device)
        weighted_loss = loss * (1.0 - rewards_tensor.mean())  # Higher loss for lower rewards
        
        # Bottleneck regularization
        # Encourage compression by penalizing long messages
        message_lengths = torch.tensor([len(msg) for msg in messages], device=device, dtype=torch.float)
        length_penalty = training_config.compression_penalty * message_lengths.mean()
        
        # Entropy regularization to encourage diversity
        token_logits = logits.view(-1, logits.size(-1))
        token_probs = F.softmax(token_logits, dim=-1)
        entropy_bonus = -training_config.entropy_regularization * torch.mean(
            torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
        )
        
        total_loss = weighted_loss + length_penalty + entropy_bonus
        
        return total_loss


def main():
    """Main function to run bottleneck experiments."""
    from experiment_config import ULTRA_TINY_CONFIG, SCALING_STUDY_CONFIG, BOTTLENECK_STUDY_CONFIG
    
    # Choose configuration
    config = ULTRA_TINY_CONFIG  # Start with ultra-tiny configuration
    
    # Enable wandb logging
    config.use_wandb = True
    
    # Run experiment
    experiment = BottleneckExperiment(config)
    results = experiment.run_full_experiment()
    
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {experiment.exp_dir}")
    
    # Print key findings
    aggregate = results['aggregate_analysis']
    print("\nKey Findings:")
    print(f"- Tested {aggregate['summary_statistics']['total_experiments']} variants")
    print(f"- Mean success rate: {aggregate['summary_statistics']['success_rate_stats']['mean']:.3f}")
    print(f"- Mean compositionality: {aggregate['summary_statistics']['compositionality_stats']['mean']:.3f}")
    print(f"- Mean convergence: {aggregate['summary_statistics']['convergence_stats']['mean_epochs']:.1f} epochs")


if __name__ == "__main__":
    main()