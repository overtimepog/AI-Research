# Emergent Communication in Tiny Multi-Agent Networks with FLARE Attention

Research implementation exploring how small models (500M-1B parameters) develop communication protocols using FLARE (Fast Low-rank Attention Routing Engine) attention mechanism.

## Key Research Questions

1. **Minimal Model Sizes**: What's the smallest model that can support meaningful emergent communication?
2. **Vocabulary Evolution**: How do shared vocabularies emerge and stabilize in multi-agent systems?
3. **FLARE Efficiency**: How does FLARE's linear complexity enable scaling to larger agent populations?

## Architecture Overview

### FLARE Attention Mechanism
- **Linear Complexity**: O(NM) instead of O(N²) through fixed-length latent sequences
- **Routing Strategy**: Attention routes through M latent tokens (M ≪ N)
- **Multi-Head Design**: Independent low-rank projections per head enable diverse communication patterns

### Agent Design
- **Model Size**: 100M-500M parameters per agent
- **Communication Protocol**: Discrete message passing with emergent vocabulary
- **Learning Paradigm**: Multi-agent reinforcement learning with communication rewards

## Project Structure

```
tiny-agent-comm/
├── src/
│   ├── agents/          # Agent implementations
│   ├── attention/       # FLARE attention modules
│   ├── communication/   # Communication protocols
│   └── environments/    # Multi-agent environments
├── experiments/         # Experiment scripts and configs
├── models/             # Trained model checkpoints
├── data/               # Communication logs and vocabularies
├── tests/              # Unit and integration tests
└── docs/               # Research documentation
```

## Key Findings

Based on recent research (2024-2025):
- Models as small as 100M parameters can develop meaningful communication protocols
- FLARE attention enables efficient scaling to million-token contexts
- Emergent vocabularies stabilize after ~10k interaction episodes
- Community-specific protocols emerge based on task requirements

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.agents import TinyAgent
from src.environments import CommunicationEnvironment

# Initialize agents with FLARE attention
agents = [TinyAgent(hidden_dim=768, num_heads=12, latent_tokens=32) for _ in range(4)]

# Create environment
env = CommunicationEnvironment(num_agents=4)

# Run communication episode
observations = env.reset()
for step in range(100):
    actions, messages = zip(*[agent.act(obs) for agent, obs in zip(agents, observations)])
    observations, rewards, done = env.step(actions, messages)
```

## References

- FLARE: Fast Low-rank Attention Routing Engine (arXiv:2508.12594)
- Multi-agent systems powered by large language models (Frontiers, 2024)
- Generative Emergent Communication frameworks (arXiv:2501.00226)