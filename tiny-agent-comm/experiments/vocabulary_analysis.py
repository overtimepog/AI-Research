"""
Tools for analyzing emergent vocabulary structure and compositionality in ultra-constrained models.

This module provides comprehensive analysis tools for understanding how communication
protocols emerge in resource-constrained environments and measuring the efficiency
and structure of emergent vocabularies.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import pickle
from pathlib import Path


@dataclass 
class VocabularySnapshot:
    """Snapshot of vocabulary state at a particular training step."""
    
    step: int
    messages: List[List[int]]  # All messages produced
    contexts: List[Dict]       # Contexts that produced each message
    rewards: List[float]       # Success rates for each message
    model_params: int          # Model parameter count
    bottleneck_strength: float # Communication bottleneck
    
    def __post_init__(self):
        """Validate and compute basic statistics."""
        assert len(self.messages) == len(self.contexts) == len(self.rewards)
        self.n_messages = len(self.messages)
        self.vocab_used = set()
        for msg in self.messages:
            self.vocab_used.update(msg)
        self.vocab_size = len(self.vocab_used)


class VocabularyAnalyzer:
    """Comprehensive analyzer for emergent communication vocabularies."""
    
    def __init__(self, vocab_size: int = 1000, max_message_length: int = 8):
        self.vocab_size = vocab_size
        self.max_message_length = max_message_length
        self.snapshots: List[VocabularySnapshot] = []
        
        # Analysis results cache
        self.analysis_cache: Dict[str, Any] = {}
        
    def add_snapshot(self, snapshot: VocabularySnapshot):
        """Add a vocabulary snapshot for analysis."""
        self.snapshots.append(snapshot)
        # Clear relevant cache entries
        self.analysis_cache.clear()
    
    def compute_entropy_metrics(self, snapshot: VocabularySnapshot) -> Dict[str, float]:
        """Compute various entropy-based metrics for vocabulary usage."""
        messages = snapshot.messages
        
        # Token-level entropy
        token_counts = Counter()
        for msg in messages:
            for token in msg:
                token_counts[token] += 1
        
        total_tokens = sum(token_counts.values())
        token_probs = np.array([count / total_tokens for count in token_counts.values()])
        token_entropy = -np.sum(token_probs * np.log2(token_probs + 1e-10))
        
        # Message-level entropy  
        message_tuples = [tuple(msg) for msg in messages]
        message_counts = Counter(message_tuples)
        message_probs = np.array([count / len(messages) for count in message_counts.values()])
        message_entropy = -np.sum(message_probs * np.log2(message_probs + 1e-10))
        
        # Position-specific entropy
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
        
        return {
            "token_entropy": float(token_entropy),
            "message_entropy": float(message_entropy),
            "mean_position_entropy": float(np.mean(position_entropies)) if position_entropies else 0.0,
            "vocab_utilization": len(snapshot.vocab_used) / self.vocab_size,
            "effective_vocab_size": len(message_counts),
            "compression_ratio": message_entropy / (self.max_message_length * np.log2(self.vocab_size))
        }
    
    def compute_compositionality_metrics(self, snapshot: VocabularySnapshot) -> Dict[str, float]:
        """Analyze compositional structure in the vocabulary."""
        messages = snapshot.messages
        contexts = snapshot.contexts
        rewards = snapshot.rewards
        
        # Group messages by success
        successful_msgs = [msg for msg, reward in zip(messages, rewards) if reward > 0.5]
        failed_msgs = [msg for msg, reward in zip(messages, rewards) if reward <= 0.5]
        
        if not successful_msgs:
            return {"compositionality_score": 0.0, "systematic_score": 0.0}
        
        # Analyze token co-occurrence patterns
        cooccurrence = self._compute_cooccurrence_matrix(successful_msgs)
        
        # Systematic generalization test
        systematic_score = self._test_systematic_generalization(messages, contexts, rewards)
        
        # Constituent swapping test  
        swap_score = self._test_constituent_swapping(successful_msgs, contexts[:len(successful_msgs)])
        
        # Hierarchical structure detection
        hierarchical_score = self._detect_hierarchical_structure(successful_msgs)
        
        # Overall compositionality score
        compositionality_score = np.mean([systematic_score, swap_score, hierarchical_score])
        
        return {
            "compositionality_score": float(compositionality_score),
            "systematic_score": float(systematic_score),
            "constituent_swap_score": float(swap_score),
            "hierarchical_score": float(hierarchical_score),
            "cooccurrence_structure": float(np.mean(cooccurrence))
        }
    
    def compute_efficiency_metrics(self, snapshot: VocabularySnapshot) -> Dict[str, float]:
        """Compute communication efficiency metrics."""
        messages = snapshot.messages
        rewards = snapshot.rewards
        
        # Success rate
        success_rate = np.mean(rewards)
        
        # Message length efficiency
        successful_msgs = [msg for msg, reward in zip(messages, rewards) if reward > 0.5]
        if successful_msgs:
            mean_length = np.mean([len(msg) for msg in successful_msgs])
            length_efficiency = success_rate / mean_length if mean_length > 0 else 0.0
        else:
            mean_length = 0.0
            length_efficiency = 0.0
        
        # Information density (bits per token)
        entropy_metrics = self.compute_entropy_metrics(snapshot)
        info_density = entropy_metrics["token_entropy"]
        
        # Communication efficiency: success per bit
        total_bits = sum(len(msg) * np.log2(self.vocab_size) for msg in messages)
        bit_efficiency = sum(rewards) / total_bits if total_bits > 0 else 0.0
        
        return {
            "success_rate": float(success_rate),
            "mean_message_length": float(mean_length), 
            "length_efficiency": float(length_efficiency),
            "information_density": float(info_density),
            "bit_efficiency": float(bit_efficiency),
            "compression_achieved": float(entropy_metrics["compression_ratio"])
        }
    
    def compute_emergence_dynamics(self) -> Dict[str, List[float]]:
        """Analyze how vocabulary emerges over time."""
        if len(self.snapshots) < 2:
            return {}
        
        dynamics = {
            "steps": [],
            "vocab_growth": [],
            "entropy_growth": [],
            "compositionality_growth": [],
            "efficiency_growth": [],
            "convergence_rate": []
        }
        
        for i, snapshot in enumerate(self.snapshots):
            entropy_metrics = self.compute_entropy_metrics(snapshot)
            compositionality_metrics = self.compute_compositionality_metrics(snapshot)
            efficiency_metrics = self.compute_efficiency_metrics(snapshot)
            
            dynamics["steps"].append(snapshot.step)
            dynamics["vocab_growth"].append(len(snapshot.vocab_used))
            dynamics["entropy_growth"].append(entropy_metrics["token_entropy"])
            dynamics["compositionality_growth"].append(compositionality_metrics["compositionality_score"])
            dynamics["efficiency_growth"].append(efficiency_metrics["length_efficiency"])
            
            # Convergence rate (change from previous snapshot)
            if i > 0:
                prev_entropy = self.compute_entropy_metrics(self.snapshots[i-1])["token_entropy"]
                convergence_rate = abs(entropy_metrics["token_entropy"] - prev_entropy)
                dynamics["convergence_rate"].append(convergence_rate)
            else:
                dynamics["convergence_rate"].append(0.0)
        
        return dynamics
    
    def analyze_parameter_scaling(self, snapshots_by_params: Dict[int, List[VocabularySnapshot]]) -> Dict[str, Any]:
        """Analyze how vocabulary properties scale with model parameters."""
        scaling_data = {
            "param_counts": [],
            "final_vocab_sizes": [],
            "final_entropies": [],
            "final_compositionality": [],
            "final_efficiency": [],
            "convergence_speeds": []
        }
        
        for param_count, snapshots in snapshots_by_params.items():
            if not snapshots:
                continue
                
            final_snapshot = snapshots[-1]
            
            # Final metrics
            entropy_metrics = self.compute_entropy_metrics(final_snapshot)
            comp_metrics = self.compute_compositionality_metrics(final_snapshot)
            eff_metrics = self.compute_efficiency_metrics(final_snapshot)
            
            # Convergence speed (steps to reach 90% of final performance)
            success_rates = [self.compute_efficiency_metrics(s)["success_rate"] for s in snapshots]
            final_success = success_rates[-1] if success_rates else 0.0
            target_success = 0.9 * final_success
            
            convergence_step = len(snapshots)  # Default to end
            for i, success_rate in enumerate(success_rates):
                if success_rate >= target_success:
                    convergence_step = i
                    break
            
            scaling_data["param_counts"].append(param_count)
            scaling_data["final_vocab_sizes"].append(len(final_snapshot.vocab_used))
            scaling_data["final_entropies"].append(entropy_metrics["token_entropy"])
            scaling_data["final_compositionality"].append(comp_metrics["compositionality_score"])
            scaling_data["final_efficiency"].append(eff_metrics["length_efficiency"])
            scaling_data["convergence_speeds"].append(convergence_step)
        
        return scaling_data
    
    def _compute_cooccurrence_matrix(self, messages: List[List[int]]) -> np.ndarray:
        """Compute token co-occurrence matrix."""
        cooccurrence = np.zeros((self.vocab_size, self.vocab_size))
        
        for msg in messages:
            for i, token1 in enumerate(msg):
                for j, token2 in enumerate(msg):
                    if i != j:
                        cooccurrence[token1, token2] += 1
        
        # Normalize
        row_sums = cooccurrence.sum(axis=1, keepdims=True)
        cooccurrence = np.divide(cooccurrence, row_sums, where=row_sums != 0)
        
        return cooccurrence
    
    def _test_systematic_generalization(self, messages: List[List[int]], 
                                      contexts: List[Dict], rewards: List[float]) -> float:
        """Test systematic generalization capability."""
        if not contexts:
            return 0.0
        
        # Group by context patterns
        context_patterns = defaultdict(list)
        for msg, ctx, reward in zip(messages, contexts, rewards):
            # Create context signature (simplified)
            signature = tuple(sorted(ctx.items())) if isinstance(ctx, dict) else str(ctx)
            context_patterns[signature].append((msg, reward))
        
        # Test generalization across similar contexts
        generalization_scores = []
        for pattern, msg_rewards in context_patterns.items():
            if len(msg_rewards) > 1:
                rewards_list = [reward for _, reward in msg_rewards]
                consistency = 1.0 - np.std(rewards_list)  # Lower std = better generalization
                generalization_scores.append(max(0.0, consistency))
        
        return np.mean(generalization_scores) if generalization_scores else 0.0
    
    def _test_constituent_swapping(self, messages: List[List[int]], contexts: List[Dict]) -> float:
        """Test if constituents can be swapped systematically."""
        if len(messages) < 4:
            return 0.0
        
        # Find potential constituent patterns
        constituent_scores = []
        
        # Look for messages that share prefixes or suffixes
        for i, msg1 in enumerate(messages[:20]):  # Limit for efficiency
            for j, msg2 in enumerate(messages[i+1:21]):
                if len(msg1) >= 2 and len(msg2) >= 2:
                    # Check prefix/suffix similarity
                    prefix_sim = int(msg1[0] == msg2[0])
                    suffix_sim = int(msg1[-1] == msg2[-1])
                    
                    if prefix_sim or suffix_sim:
                        constituent_scores.append((prefix_sim + suffix_sim) / 2)
        
        return np.mean(constituent_scores) if constituent_scores else 0.0
    
    def _detect_hierarchical_structure(self, messages: List[List[int]]) -> float:
        """Detect hierarchical structure in vocabulary."""
        if len(messages) < 10:
            return 0.0
        
        # Build message graph based on shared tokens
        G = nx.Graph()
        
        # Add messages as nodes
        for i, msg in enumerate(messages):
            G.add_node(i, message=msg)
        
        # Add edges based on token overlap
        for i, msg1 in enumerate(messages):
            for j, msg2 in enumerate(messages[i+1:], i+1):
                overlap = len(set(msg1) & set(msg2))
                if overlap > 0:
                    G.add_edge(i, j, weight=overlap / max(len(msg1), len(msg2)))
        
        # Analyze graph structure
        if len(G.nodes) > 1:
            try:
                clustering = nx.clustering(G)
                avg_clustering = np.mean(list(clustering.values()))
                return float(avg_clustering)
            except:
                return 0.0
        
        return 0.0
    
    def visualize_vocabulary_evolution(self, save_path: Optional[Path] = None):
        """Create visualizations of vocabulary evolution."""
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots for evolution visualization")
            return
        
        dynamics = self.compute_emergence_dynamics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Vocabulary Evolution Analysis', fontsize=16)
        
        # Vocabulary growth
        axes[0,0].plot(dynamics["steps"], dynamics["vocab_growth"])
        axes[0,0].set_title('Vocabulary Size Growth')
        axes[0,0].set_xlabel('Training Steps')
        axes[0,0].set_ylabel('Unique Tokens Used')
        
        # Entropy evolution
        axes[0,1].plot(dynamics["steps"], dynamics["entropy_growth"])
        axes[0,1].set_title('Entropy Evolution')
        axes[0,1].set_xlabel('Training Steps')
        axes[0,1].set_ylabel('Token Entropy (bits)')
        
        # Compositionality development
        axes[1,0].plot(dynamics["steps"], dynamics["compositionality_growth"])
        axes[1,0].set_title('Compositionality Development')
        axes[1,0].set_xlabel('Training Steps') 
        axes[1,0].set_ylabel('Compositionality Score')
        
        # Efficiency evolution
        axes[1,1].plot(dynamics["steps"], dynamics["efficiency_growth"])
        axes[1,1].set_title('Communication Efficiency')
        axes[1,1].set_xlabel('Training Steps')
        axes[1,1].set_ylabel('Length Efficiency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if not self.snapshots:
            return {"error": "No snapshots available for analysis"}
        
        final_snapshot = self.snapshots[-1]
        
        # Compute all metrics
        entropy_metrics = self.compute_entropy_metrics(final_snapshot)
        comp_metrics = self.compute_compositionality_metrics(final_snapshot)
        eff_metrics = self.compute_efficiency_metrics(final_snapshot)
        dynamics = self.compute_emergence_dynamics()
        
        report = {
            "experiment_summary": {
                "total_snapshots": len(self.snapshots),
                "final_step": final_snapshot.step,
                "model_params": final_snapshot.model_params,
                "bottleneck_strength": final_snapshot.bottleneck_strength,
            },
            "final_metrics": {
                **entropy_metrics,
                **comp_metrics, 
                **eff_metrics
            },
            "emergence_dynamics": dynamics,
            "key_findings": self._extract_key_findings(entropy_metrics, comp_metrics, eff_metrics, dynamics)
        }
        
        return report
    
    def _extract_key_findings(self, entropy_metrics: Dict, comp_metrics: Dict, 
                            eff_metrics: Dict, dynamics: Dict) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        # Vocabulary utilization
        util = entropy_metrics["vocab_utilization"]
        if util < 0.1:
            findings.append("Very sparse vocabulary usage (<10% of available tokens)")
        elif util > 0.8:
            findings.append("High vocabulary utilization (>80% of available tokens)")
        
        # Compression achievement
        compression = entropy_metrics["compression_ratio"] 
        if compression < 0.3:
            findings.append("Strong compression achieved (< 30% of theoretical maximum)")
        elif compression > 0.8:
            findings.append("Limited compression (> 80% of theoretical maximum)")
        
        # Compositionality
        comp_score = comp_metrics["compositionality_score"]
        if comp_score > 0.7:
            findings.append("Strong compositional structure emerged")
        elif comp_score < 0.3:
            findings.append("Limited compositional structure detected")
        
        # Efficiency
        eff_score = eff_metrics["length_efficiency"]
        if eff_score > 1.0:
            findings.append("High communication efficiency achieved")
        elif eff_score < 0.5:
            findings.append("Low communication efficiency")
        
        # Convergence
        if dynamics.get("convergence_rate"):
            final_convergence = dynamics["convergence_rate"][-1]
            if final_convergence < 0.01:
                findings.append("Protocol appears to have converged")
            else:
                findings.append("Protocol still evolving at end of training")
        
        return findings
    
    def save_analysis(self, filepath: Path):
        """Save complete analysis to file."""
        analysis_data = {
            "snapshots": self.snapshots,
            "analysis_cache": self.analysis_cache,
            "config": {
                "vocab_size": self.vocab_size,
                "max_message_length": self.max_message_length
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(analysis_data, f)
    
    @classmethod
    def load_analysis(cls, filepath: Path) -> 'VocabularyAnalyzer':
        """Load saved analysis from file."""
        with open(filepath, 'rb') as f:
            analysis_data = pickle.load(f)
        
        analyzer = cls(
            vocab_size=analysis_data["config"]["vocab_size"],
            max_message_length=analysis_data["config"]["max_message_length"]
        )
        analyzer.snapshots = analysis_data["snapshots"]
        analyzer.analysis_cache = analysis_data["analysis_cache"]
        
        return analyzer


def compare_vocabulary_analyses(analyzers: List[VocabularyAnalyzer], 
                              labels: List[str]) -> Dict[str, Any]:
    """Compare vocabulary analyses across different experimental conditions."""
    comparison = {
        "conditions": labels,
        "final_metrics_comparison": {},
        "scaling_comparison": {},
        "emergence_speed_comparison": {}
    }
    
    # Compare final metrics
    final_metrics = {}
    for label, analyzer in zip(labels, analyzers):
        if analyzer.snapshots:
            final_snapshot = analyzer.snapshots[-1]
            entropy_metrics = analyzer.compute_entropy_metrics(final_snapshot)
            comp_metrics = analyzer.compute_compositionality_metrics(final_snapshot)
            eff_metrics = analyzer.compute_efficiency_metrics(final_snapshot)
            
            final_metrics[label] = {**entropy_metrics, **comp_metrics, **eff_metrics}
    
    comparison["final_metrics_comparison"] = final_metrics
    
    # Compare emergence dynamics
    emergence_comparison = {}
    for label, analyzer in zip(labels, analyzers):
        dynamics = analyzer.compute_emergence_dynamics()
        if dynamics:
            # Find step to reach 50% of final compositionality
            comp_scores = dynamics.get("compositionality_growth", [])
            if comp_scores:
                final_comp = comp_scores[-1]
                target_comp = 0.5 * final_comp
                emergence_step = len(comp_scores)
                for i, score in enumerate(comp_scores):
                    if score >= target_comp:
                        emergence_step = i
                        break
                emergence_comparison[label] = emergence_step
    
    comparison["emergence_speed_comparison"] = emergence_comparison
    
    return comparison