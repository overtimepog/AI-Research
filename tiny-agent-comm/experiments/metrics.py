"""
Metrics for measuring communication efficiency and protocol emergence.

This module provides comprehensive metrics for evaluating emergent communication
protocols, including information-theoretic measures, efficiency metrics, and
bottleneck analysis tools.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import math


@dataclass
class MessageBatch:
    """Batch of messages with associated metadata."""
    messages: List[List[int]]
    contexts: List[Dict[str, Any]]
    rewards: List[float]
    speaker_ids: Optional[List[int]] = None
    listener_ids: Optional[List[int]] = None
    timestamps: Optional[List[int]] = None


class CommunicationMetrics:
    """Comprehensive metrics for analyzing emergent communication."""
    
    def __init__(self, vocab_size: int = 1000, max_message_length: int = 8):
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        
        # Cached computations
        self._cache = {}
        self._cache_keys = set()
    
    def clear_cache(self):
        """Clear cached computations."""
        self._cache.clear()
        self._cache_keys.clear()
    
    def compute_basic_metrics(self, batch: MessageBatch) -> Dict[str, float]:
        """Compute basic communication metrics."""
        messages = batch.messages
        rewards = batch.rewards
        
        if not messages:
            return {"success_rate": 0.0, "avg_message_length": 0.0, "vocab_usage": 0.0}
        
        # Success rate
        success_rate = np.mean(rewards)
        
        # Message length statistics
        lengths = [len(msg) for msg in messages]
        avg_length = np.mean(lengths)
        length_std = np.std(lengths)
        
        # Vocabulary usage
        unique_tokens = set()
        for msg in messages:
            unique_tokens.update(msg)
        vocab_usage = len(unique_tokens) / self.vocab_size
        
        # Message diversity
        unique_messages = len(set(tuple(msg) for msg in messages))
        message_diversity = unique_messages / len(messages)
        
        return {
            "success_rate": float(success_rate),
            "avg_message_length": float(avg_length),
            "length_std": float(length_std),
            "vocab_usage": float(vocab_usage),
            "message_diversity": float(message_diversity),
            "unique_tokens": len(unique_tokens),
            "unique_messages": unique_messages
        }
    
    def compute_entropy_metrics(self, batch: MessageBatch) -> Dict[str, float]:
        """Compute various entropy measures for communication analysis."""
        messages = batch.messages
        
        if not messages:
            return {"token_entropy": 0.0, "message_entropy": 0.0, "conditional_entropy": 0.0}
        
        # Token-level entropy
        token_counts = Counter()
        total_tokens = 0
        for msg in messages:
            for token in msg:
                token_counts[token] += 1
                total_tokens += 1
        
        if total_tokens == 0:
            return {"token_entropy": 0.0, "message_entropy": 0.0, "conditional_entropy": 0.0}
        
        token_probs = np.array([count / total_tokens for count in token_counts.values()])
        token_entropy = -np.sum(token_probs * np.log2(token_probs + 1e-10))
        
        # Message-level entropy
        message_tuples = [tuple(msg) for msg in messages]
        message_counts = Counter(message_tuples)
        message_probs = np.array([count / len(messages) for count in message_counts.values()])
        message_entropy = -np.sum(message_probs * np.log2(message_probs + 1e-10))
        
        # Position-specific entropy (conditional entropy)
        position_entropies = []
        for pos in range(self.max_message_length):
            pos_tokens = []
            for msg in messages:
                if pos < len(msg):
                    pos_tokens.append(msg[pos])
            if pos_tokens:
                pos_counts = Counter(pos_tokens)
                pos_probs = np.array([count / len(pos_tokens) for count in pos_counts.values()])
                pos_entropy = -np.sum(pos_probs * np.log2(pos_probs + 1e-10))
                position_entropies.append(pos_entropy)
        
        conditional_entropy = np.mean(position_entropies) if position_entropies else 0.0
        
        # Perplexity measures
        token_perplexity = 2 ** token_entropy
        message_perplexity = 2 ** message_entropy
        
        return {
            "token_entropy": float(token_entropy),
            "message_entropy": float(message_entropy), 
            "conditional_entropy": float(conditional_entropy),
            "token_perplexity": float(token_perplexity),
            "message_perplexity": float(message_perplexity),
            "entropy_rate": float(token_entropy / np.mean([len(msg) for msg in messages]) if messages else 0)
        }
    
    def compute_mutual_information(self, batch: MessageBatch) -> Dict[str, float]:
        """Compute mutual information between messages and contexts/outcomes."""
        messages = batch.messages
        contexts = batch.contexts
        rewards = batch.rewards
        
        if len(messages) != len(contexts) or len(messages) != len(rewards):
            return {"mi_message_context": 0.0, "mi_message_reward": 0.0}
        
        # Convert messages to discrete features
        message_features = []
        for msg in messages:
            # Use hash of message tuple as feature
            msg_hash = hash(tuple(msg)) % 10000  # Mod to keep reasonable range
            message_features.append(msg_hash)
        
        # Convert contexts to discrete features (simplified)
        context_features = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                # Hash relevant context properties
                ctx_str = str(sorted(ctx.items()))
                ctx_hash = hash(ctx_str) % 10000
                context_features.append(ctx_hash)
            else:
                context_features.append(hash(str(ctx)) % 10000)
        
        # Discretize rewards
        reward_bins = np.digitize(rewards, bins=np.linspace(0, 1, 11))
        
        # Compute mutual information
        try:
            mi_message_context = mutual_info_score(message_features, context_features)
            mi_message_reward = mutual_info_score(message_features, reward_bins)
            
            # Normalized versions
            h_message = self._discrete_entropy(message_features)
            h_context = self._discrete_entropy(context_features)
            h_reward = self._discrete_entropy(reward_bins)
            
            nmi_message_context = mi_message_context / (h_message + h_context + 1e-10)
            nmi_message_reward = mi_message_reward / (h_message + h_reward + 1e-10)
            
        except Exception as e:
            return {"mi_message_context": 0.0, "mi_message_reward": 0.0}
        
        return {
            "mi_message_context": float(mi_message_context),
            "mi_message_reward": float(mi_message_reward),
            "nmi_message_context": float(nmi_message_context),
            "nmi_message_reward": float(nmi_message_reward)
        }
    
    def compute_efficiency_metrics(self, batch: MessageBatch) -> Dict[str, float]:
        """Compute communication efficiency metrics."""
        messages = batch.messages
        rewards = batch.rewards
        
        if not messages:
            return {"length_efficiency": 0.0, "bit_efficiency": 0.0, "success_per_token": 0.0}
        
        # Basic efficiency measures
        total_success = sum(rewards)
        total_tokens = sum(len(msg) for msg in messages)
        total_bits = sum(len(msg) * math.log2(self.vocab_size) for msg in messages)
        
        length_efficiency = total_success / total_tokens if total_tokens > 0 else 0.0
        bit_efficiency = total_success / total_bits if total_bits > 0 else 0.0
        success_per_token = length_efficiency
        
        # Message-specific efficiency
        message_efficiencies = []
        for msg, reward in zip(messages, rewards):
            if len(msg) > 0:
                msg_efficiency = reward / len(msg)
                message_efficiencies.append(msg_efficiency)
        
        avg_message_efficiency = np.mean(message_efficiencies) if message_efficiencies else 0.0
        efficiency_std = np.std(message_efficiencies) if message_efficiencies else 0.0
        
        # Pareto efficiency: proportion of messages on efficiency frontier
        pareto_efficient_count = 0
        for i, (msg1, reward1) in enumerate(zip(messages, rewards)):
            is_pareto = True
            for j, (msg2, reward2) in enumerate(zip(messages, rewards)):
                if i != j:
                    # Dominated if other message is shorter AND better reward
                    if len(msg2) <= len(msg1) and reward2 >= reward1 and (len(msg2) < len(msg1) or reward2 > reward1):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_efficient_count += 1
        
        pareto_efficiency = pareto_efficient_count / len(messages) if messages else 0.0
        
        return {
            "length_efficiency": float(length_efficiency),
            "bit_efficiency": float(bit_efficiency),
            "success_per_token": float(success_per_token),
            "avg_message_efficiency": float(avg_message_efficiency),
            "efficiency_std": float(efficiency_std),
            "pareto_efficiency": float(pareto_efficiency)
        }
    
    def compute_compositionality_metrics(self, batch: MessageBatch) -> Dict[str, float]:
        """Compute metrics measuring compositional structure in messages."""
        messages = batch.messages
        contexts = batch.contexts
        rewards = batch.rewards
        
        if len(messages) < 10:  # Need sufficient data
            return {"compositionality_score": 0.0, "systematicity": 0.0, "productivity": 0.0}
        
        # Positional consistency: do tokens in same positions have consistent meanings?
        positional_consistency = self._compute_positional_consistency(messages, contexts, rewards)
        
        # Compositional generalization: can meanings be combined systematically?
        compositional_generalization = self._compute_compositional_generalization(messages, contexts, rewards)
        
        # Productivity: can new messages be formed from existing components?
        productivity = self._compute_productivity(messages)
        
        # Systematicity: are similar contexts mapped to similar messages?
        systematicity = self._compute_systematicity(messages, contexts)
        
        # Overall compositionality score
        compositionality_score = np.mean([
            positional_consistency,
            compositional_generalization,
            productivity,
            systematicity
        ])
        
        return {
            "compositionality_score": float(compositionality_score),
            "positional_consistency": float(positional_consistency),
            "compositional_generalization": float(compositional_generalization),
            "productivity": float(productivity),
            "systematicity": float(systematicity)
        }
    
    def compute_convergence_metrics(self, message_batches: List[MessageBatch]) -> Dict[str, float]:
        """Compute metrics measuring protocol convergence over time."""
        if len(message_batches) < 2:
            return {"convergence_rate": 0.0, "stability": 0.0, "consensus": 0.0}
        
        # Track vocabulary stability
        vocab_stability_scores = []
        for i in range(1, len(message_batches)):
            prev_vocab = set()
            curr_vocab = set()
            
            for msg in message_batches[i-1].messages:
                prev_vocab.update(msg)
            for msg in message_batches[i].messages:
                curr_vocab.update(msg)
            
            if prev_vocab or curr_vocab:
                jaccard = len(prev_vocab & curr_vocab) / len(prev_vocab | curr_vocab)
                vocab_stability_scores.append(jaccard)
        
        vocab_stability = np.mean(vocab_stability_scores) if vocab_stability_scores else 0.0
        
        # Track message protocol convergence
        protocol_convergence_scores = []
        for i in range(1, len(message_batches)):
            prev_messages = set(tuple(msg) for msg in message_batches[i-1].messages)
            curr_messages = set(tuple(msg) for msg in message_batches[i].messages)
            
            if prev_messages or curr_messages:
                jaccard = len(prev_messages & curr_messages) / len(prev_messages | curr_messages)
                protocol_convergence_scores.append(jaccard)
        
        protocol_convergence = np.mean(protocol_convergence_scores) if protocol_convergence_scores else 0.0
        
        # Success rate stability
        success_rates = [np.mean(batch.rewards) for batch in message_batches]
        success_stability = 1.0 - np.std(success_rates[-5:]) if len(success_rates) >= 5 else 0.0
        
        # Overall convergence metrics
        convergence_rate = np.mean([vocab_stability, protocol_convergence])
        stability = success_stability
        consensus = protocol_convergence
        
        return {
            "convergence_rate": float(convergence_rate),
            "stability": float(stability),
            "consensus": float(consensus),
            "vocab_stability": float(vocab_stability),
            "protocol_convergence": float(protocol_convergence)
        }
    
    def _discrete_entropy(self, values: List[int]) -> float:
        """Compute entropy of discrete values."""
        if not values:
            return 0.0
        
        counts = Counter(values)
        total = len(values)
        probs = np.array([count / total for count in counts.values()])
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _compute_positional_consistency(self, messages: List[List[int]], 
                                      contexts: List[Dict], rewards: List[float]) -> float:
        """Compute how consistently tokens in same positions relate to contexts."""
        if not messages or not contexts:
            return 0.0
        
        position_consistencies = []
        
        for pos in range(self.max_message_length):
            pos_token_contexts = defaultdict(list)
            
            # Group contexts by token at this position
            for msg, ctx in zip(messages, contexts):
                if pos < len(msg):
                    token = msg[pos]
                    pos_token_contexts[token].append(ctx)
            
            # Measure consistency within each token's contexts
            token_consistencies = []
            for token, token_contexts in pos_token_contexts.items():
                if len(token_contexts) > 1:
                    # Simplified: measure variation in context hashes
                    ctx_hashes = [hash(str(sorted(ctx.items()))) if isinstance(ctx, dict) else hash(str(ctx)) 
                                 for ctx in token_contexts]
                    unique_hashes = len(set(ctx_hashes))
                    consistency = 1.0 - (unique_hashes - 1) / (len(ctx_hashes) - 1)
                    token_consistencies.append(max(0.0, consistency))
            
            if token_consistencies:
                position_consistencies.append(np.mean(token_consistencies))
        
        return np.mean(position_consistencies) if position_consistencies else 0.0
    
    def _compute_compositional_generalization(self, messages: List[List[int]], 
                                           contexts: List[Dict], rewards: List[float]) -> float:
        """Measure ability to generalize compositionally to new combinations."""
        if len(messages) < 20:
            return 0.0
        
        # Group messages by success
        successful_msgs = [(msg, ctx) for msg, ctx, reward in zip(messages, contexts, rewards) if reward > 0.5]
        failed_msgs = [(msg, ctx) for msg, ctx, reward in zip(messages, contexts, rewards) if reward <= 0.5]
        
        if len(successful_msgs) < 10:
            return 0.0
        
        # Look for compositional patterns in successful messages
        # Simplified: check if message parts can be recombined successfully
        component_success = defaultdict(list)
        
        for msg, ctx in successful_msgs:
            if len(msg) >= 2:
                # Track success of message components
                for i in range(len(msg)):
                    component = msg[i]
                    component_success[component].append(1.0)
        
        # Average success rate of components
        component_avg_success = {}
        for component, successes in component_success.items():
            component_avg_success[component] = np.mean(successes)
        
        # Predict success of unseen combinations
        generalization_scores = []
        for msg, ctx in successful_msgs[:10]:  # Test on subset
            if len(msg) >= 2:
                predicted_success = np.mean([component_avg_success.get(token, 0.5) for token in msg])
                actual_success = 1.0  # These are successful messages
                generalization_scores.append(1.0 - abs(predicted_success - actual_success))
        
        return np.mean(generalization_scores) if generalization_scores else 0.0
    
    def _compute_productivity(self, messages: List[List[int]]) -> float:
        """Measure productivity: ability to create new messages from existing components."""
        if len(messages) < 10:
            return 0.0
        
        # Get all tokens used
        all_tokens = set()
        for msg in messages:
            all_tokens.update(msg)
        
        if len(all_tokens) < 2:
            return 0.0
        
        # Count possible vs actual combinations
        unique_messages = set(tuple(msg) for msg in messages)
        
        # Estimate potential combinations (simplified)
        avg_msg_length = np.mean([len(msg) for msg in messages])
        potential_combinations = len(all_tokens) ** avg_msg_length
        
        # Productivity score
        productivity = min(1.0, len(unique_messages) / potential_combinations)
        
        return float(productivity)
    
    def _compute_systematicity(self, messages: List[List[int]], contexts: List[Dict]) -> float:
        """Measure systematicity: similar contexts should produce similar messages."""
        if len(messages) < 10 or len(contexts) < 10:
            return 0.0
        
        systematicity_scores = []
        
        # Compare pairs of context-message mappings
        for i in range(min(20, len(messages))):
            for j in range(i+1, min(20, len(messages))):
                ctx1, ctx2 = contexts[i], contexts[j]
                msg1, msg2 = messages[i], messages[j]
                
                # Measure context similarity (simplified)
                if isinstance(ctx1, dict) and isinstance(ctx2, dict):
                    ctx1_items = set(str(k) + str(v) for k, v in ctx1.items())
                    ctx2_items = set(str(k) + str(v) for k, v in ctx2.items())
                    if ctx1_items or ctx2_items:
                        ctx_similarity = len(ctx1_items & ctx2_items) / len(ctx1_items | ctx2_items)
                    else:
                        ctx_similarity = 0.0
                else:
                    ctx_similarity = 1.0 if str(ctx1) == str(ctx2) else 0.0
                
                # Measure message similarity
                msg1_set = set(msg1)
                msg2_set = set(msg2)
                if msg1_set or msg2_set:
                    msg_similarity = len(msg1_set & msg2_set) / len(msg1_set | msg2_set)
                else:
                    msg_similarity = 0.0
                
                # Systematicity: high context similarity should correlate with high message similarity
                if ctx_similarity > 0.1:  # Only consider somewhat similar contexts
                    systematicity_scores.append(msg_similarity)
        
        return np.mean(systematicity_scores) if systematicity_scores else 0.0


class BottleneckMetrics:
    """Metrics specifically for analyzing information bottleneck effects."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
    
    def compute_compression_metrics(self, batch: MessageBatch) -> Dict[str, float]:
        """Compute information compression metrics."""
        messages = batch.messages
        
        if not messages:
            return {"compression_ratio": 0.0, "rate_distortion": 0.0}
        
        # Theoretical maximum information
        max_bits_per_token = math.log2(self.vocab_size)
        max_info = max_bits_per_token * max(len(msg) for msg in messages) if messages else 0
        
        # Actual information content (entropy-based)
        token_counts = Counter()
        total_tokens = 0
        for msg in messages:
            for token in msg:
                token_counts[token] += 1
                total_tokens += 1
        
        if total_tokens == 0:
            return {"compression_ratio": 0.0, "rate_distortion": 0.0}
        
        token_probs = np.array([count / total_tokens for count in token_counts.values()])
        actual_entropy = -np.sum(token_probs * np.log2(token_probs + 1e-10))
        
        # Compression ratio
        compression_ratio = actual_entropy / max_bits_per_token if max_bits_per_token > 0 else 0.0
        
        # Rate-distortion approximation
        avg_message_length = np.mean([len(msg) for msg in messages])
        rate = actual_entropy  # bits per token
        distortion = 1.0 - np.mean(batch.rewards)  # inverse of success rate
        rate_distortion = rate / (1.0 + distortion) if distortion < 1.0 else rate
        
        # Information density
        info_density = actual_entropy / avg_message_length if avg_message_length > 0 else 0.0
        
        return {
            "compression_ratio": float(compression_ratio),
            "rate_distortion": float(rate_distortion),
            "info_density": float(info_density),
            "actual_entropy": float(actual_entropy),
            "max_entropy": float(max_bits_per_token)
        }
    
    def compute_bottleneck_pressure(self, batch: MessageBatch, max_message_length: int) -> Dict[str, float]:
        """Compute metrics indicating bottleneck pressure effects."""
        messages = batch.messages
        rewards = batch.rewards
        
        if not messages:
            return {"length_pressure": 0.0, "vocab_pressure": 0.0, "efficiency_pressure": 0.0}
        
        # Length pressure: how close are messages to the maximum length?
        lengths = [len(msg) for msg in messages]
        avg_length = np.mean(lengths)
        length_pressure = avg_length / max_message_length
        
        # Vocabulary pressure: how concentrated is vocabulary usage?
        token_counts = Counter()
        for msg in messages:
            for token in msg:
                token_counts[token] += 1
        
        if token_counts:
            # Gini coefficient for vocabulary concentration
            counts = sorted(token_counts.values())
            n = len(counts)
            gini = (2 * sum((i + 1) * count for i, count in enumerate(counts)) / sum(counts) - n - 1) / n
            vocab_pressure = gini
        else:
            vocab_pressure = 0.0
        
        # Efficiency pressure: relationship between length and success
        if len(lengths) == len(rewards):
            length_reward_corr = np.corrcoef(lengths, rewards)[0, 1] if len(set(lengths)) > 1 else 0.0
            efficiency_pressure = -length_reward_corr if not np.isnan(length_reward_corr) else 0.0
        else:
            efficiency_pressure = 0.0
        
        return {
            "length_pressure": float(length_pressure),
            "vocab_pressure": float(vocab_pressure),
            "efficiency_pressure": float(efficiency_pressure)
        }
    
    def compute_emergent_structure(self, batch: MessageBatch) -> Dict[str, float]:
        """Analyze emergent structure in bottleneck-constrained communication."""
        messages = batch.messages
        contexts = batch.contexts
        
        if len(messages) < 10:
            return {"structural_complexity": 0.0, "hierarchical_organization": 0.0}
        
        # Build co-occurrence graph
        G = nx.Graph()
        
        # Add tokens as nodes
        all_tokens = set()
        for msg in messages:
            all_tokens.update(msg)
        G.add_nodes_from(all_tokens)
        
        # Add edges based on co-occurrence
        for msg in messages:
            for i in range(len(msg)):
                for j in range(i+1, len(msg)):
                    token1, token2 = msg[i], msg[j]
                    if G.has_edge(token1, token2):
                        G[token1][token2]['weight'] += 1
                    else:
                        G.add_edge(token1, token2, weight=1)
        
        # Structural complexity measures
        if len(G.nodes) > 1:
            try:
                # Clustering coefficient
                clustering = nx.average_clustering(G)
                
                # Modularity (community structure)
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = nx_comm.greedy_modularity_communities(G)
                    modularity = nx_comm.modularity(G, communities)
                except:
                    modularity = 0.0
                
                # Small-world properties
                try:
                    avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0.0
                    small_world = clustering / (avg_path_length + 1e-10)
                except:
                    small_world = 0.0
                
                structural_complexity = np.mean([clustering, modularity, small_world])
                hierarchical_organization = modularity
                
            except:
                structural_complexity = 0.0
                hierarchical_organization = 0.0
        else:
            structural_complexity = 0.0
            hierarchical_organization = 0.0
        
        return {
            "structural_complexity": float(structural_complexity),
            "hierarchical_organization": float(hierarchical_organization),
            "graph_clustering": float(clustering) if 'clustering' in locals() else 0.0,
            "graph_modularity": float(modularity) if 'modularity' in locals() else 0.0
        }


class MetricsAggregator:
    """Aggregates and analyzes metrics across experiments."""
    
    def __init__(self):
        self.communication_metrics = CommunicationMetrics()
        self.bottleneck_metrics = BottleneckMetrics()
        self.results_history = []
    
    def add_experiment_results(self, experiment_id: str, batches: List[MessageBatch], 
                             config: Dict[str, Any]):
        """Add results from a complete experiment."""
        # Compute all metrics for each batch
        batch_results = []
        for batch in batches:
            batch_result = {
                "basic_metrics": self.communication_metrics.compute_basic_metrics(batch),
                "entropy_metrics": self.communication_metrics.compute_entropy_metrics(batch),
                "efficiency_metrics": self.communication_metrics.compute_efficiency_metrics(batch),
                "compositionality_metrics": self.communication_metrics.compute_compositionality_metrics(batch),
                "mutual_info_metrics": self.communication_metrics.compute_mutual_information(batch),
                "compression_metrics": self.bottleneck_metrics.compute_compression_metrics(batch),
                "bottleneck_pressure": self.bottleneck_metrics.compute_bottleneck_pressure(
                    batch, config.get('max_message_length', 8)
                ),
                "emergent_structure": self.bottleneck_metrics.compute_emergent_structure(batch)
            }
            batch_results.append(batch_result)
        
        # Compute convergence metrics across batches
        convergence_metrics = self.communication_metrics.compute_convergence_metrics(batches)
        
        # Store complete experiment results
        experiment_result = {
            "experiment_id": experiment_id,
            "config": config,
            "batch_results": batch_results,
            "convergence_metrics": convergence_metrics,
            "summary_metrics": self._compute_summary_metrics(batch_results)
        }
        
        self.results_history.append(experiment_result)
        return experiment_result
    
    def _compute_summary_metrics(self, batch_results: List[Dict]) -> Dict[str, float]:
        """Compute summary metrics across all batches in experiment."""
        if not batch_results:
            return {}
        
        summary = {}
        
        # Average metrics across batches
        metric_categories = ["basic_metrics", "entropy_metrics", "efficiency_metrics", 
                           "compositionality_metrics", "mutual_info_metrics", 
                           "compression_metrics", "bottleneck_pressure", "emergent_structure"]
        
        for category in metric_categories:
            category_metrics = {}
            for metric_name in batch_results[0][category].keys():
                values = [batch[category][metric_name] for batch in batch_results 
                         if metric_name in batch[category]]
                if values:
                    category_metrics[f"{metric_name}_mean"] = np.mean(values)
                    category_metrics[f"{metric_name}_std"] = np.std(values)
                    category_metrics[f"{metric_name}_final"] = values[-1]
            summary[category] = category_metrics
        
        return summary
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple experiments."""
        experiments = [exp for exp in self.results_history if exp["experiment_id"] in experiment_ids]
        
        if len(experiments) < 2:
            return {"error": "Need at least 2 experiments for comparison"}
        
        comparison = {
            "experiment_ids": experiment_ids,
            "parameter_effects": {},
            "bottleneck_effects": {},
            "statistical_tests": {}
        }
        
        # Extract key metrics for comparison
        param_counts = []
        bottleneck_strengths = []
        final_success_rates = []
        final_compositionality = []
        final_compression = []
        
        for exp in experiments:
            config = exp["config"]
            summary = exp["summary_metrics"]
            
            param_counts.append(config.get("param_count", 0))
            bottleneck_strengths.append(config.get("bottleneck_strength", 1.0))
            
            success_rate = summary.get("basic_metrics", {}).get("success_rate_final", 0.0)
            compositionality = summary.get("compositionality_metrics", {}).get("compositionality_score_final", 0.0)
            compression = summary.get("compression_metrics", {}).get("compression_ratio_final", 0.0)
            
            final_success_rates.append(success_rate)
            final_compositionality.append(compositionality)
            final_compression.append(compression)
        
        # Statistical analysis
        if len(param_counts) > 2:
            # Correlations
            param_success_corr = np.corrcoef(param_counts, final_success_rates)[0, 1]
            param_comp_corr = np.corrcoef(param_counts, final_compositionality)[0, 1]
            
            bottleneck_success_corr = np.corrcoef(bottleneck_strengths, final_success_rates)[0, 1]
            bottleneck_comp_corr = np.corrcoef(bottleneck_strengths, final_compositionality)[0, 1]
            bottleneck_compression_corr = np.corrcoef(bottleneck_strengths, final_compression)[0, 1]
            
            comparison["parameter_effects"] = {
                "param_success_correlation": float(param_success_corr) if not np.isnan(param_success_corr) else 0.0,
                "param_compositionality_correlation": float(param_comp_corr) if not np.isnan(param_comp_corr) else 0.0
            }
            
            comparison["bottleneck_effects"] = {
                "bottleneck_success_correlation": float(bottleneck_success_corr) if not np.isnan(bottleneck_success_corr) else 0.0,
                "bottleneck_compositionality_correlation": float(bottleneck_comp_corr) if not np.isnan(bottleneck_comp_corr) else 0.0,
                "bottleneck_compression_correlation": float(bottleneck_compression_corr) if not np.isnan(bottleneck_compression_corr) else 0.0
            }
        
        return comparison
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report across all experiments."""
        if not self.results_history:
            return {"error": "No experiment results available"}
        
        report = {
            "total_experiments": len(self.results_history),
            "experiment_summary": {},
            "key_findings": [],
            "recommendations": []
        }
        
        # Aggregate metrics across all experiments
        all_success_rates = []
        all_compositionality_scores = []
        all_compression_ratios = []
        all_param_counts = []
        all_bottleneck_strengths = []
        
        for exp in self.results_history:
            config = exp["config"]
            summary = exp["summary_metrics"]
            
            all_param_counts.append(config.get("param_count", 0))
            all_bottleneck_strengths.append(config.get("bottleneck_strength", 1.0))
            
            success_rate = summary.get("basic_metrics", {}).get("success_rate_final", 0.0)
            compositionality = summary.get("compositionality_metrics", {}).get("compositionality_score_final", 0.0)
            compression = summary.get("compression_metrics", {}).get("compression_ratio_final", 0.0)
            
            all_success_rates.append(success_rate)
            all_compositionality_scores.append(compositionality)
            all_compression_ratios.append(compression)
        
        # Summary statistics
        report["experiment_summary"] = {
            "success_rate_stats": {
                "mean": np.mean(all_success_rates),
                "std": np.std(all_success_rates),
                "min": np.min(all_success_rates),
                "max": np.max(all_success_rates)
            },
            "compositionality_stats": {
                "mean": np.mean(all_compositionality_scores),
                "std": np.std(all_compositionality_scores),
                "min": np.min(all_compositionality_scores),
                "max": np.max(all_compositionality_scores)
            },
            "compression_stats": {
                "mean": np.mean(all_compression_ratios),
                "std": np.std(all_compression_ratios),
                "min": np.min(all_compression_ratios),
                "max": np.max(all_compression_ratios)
            }
        }
        
        # Generate findings based on data
        mean_success = np.mean(all_success_rates)
        mean_comp = np.mean(all_compositionality_scores)
        mean_compression = np.mean(all_compression_ratios)
        
        if mean_success > 0.7:
            report["key_findings"].append("High overall success rates achieved across experiments")
        if mean_comp > 0.5:
            report["key_findings"].append("Significant compositional structure emerged")
        if mean_compression < 0.5:
            report["key_findings"].append("Strong information compression observed")
        
        # Parameter scaling effects
        if len(set(all_param_counts)) > 2:
            param_success_corr = np.corrcoef(all_param_counts, all_success_rates)[0, 1]
            if not np.isnan(param_success_corr):
                if param_success_corr > 0.5:
                    report["key_findings"].append("Larger models show better communication success")
                elif param_success_corr < -0.5:
                    report["key_findings"].append("Smaller models show better communication success")
        
        # Bottleneck effects
        if len(set(all_bottleneck_strengths)) > 2:
            bottleneck_comp_corr = np.corrcoef(all_bottleneck_strengths, all_compositionality_scores)[0, 1]
            if not np.isnan(bottleneck_comp_corr) and bottleneck_comp_corr < -0.3:
                report["key_findings"].append("Stronger bottlenecks increase compositional structure")
        
        return report