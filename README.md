# LPG Environments for Gymnasium

Gymnasium-compliant reinforcement learning environments from the
**Learning Progress Graphs (LPG)** paper
([Oh et al., 2020](https://arxiv.org/abs/2007.08794)).

## Installation

```bash
pip install -e .
```

---

## Quick Start

```python
import gymnasium
import lpg_envs  # registers all 15 environments

env = gymnasium.make("lpg_envs/TabularGridWorld-Dense-v0")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

---

## Play

```bash
python play.py <env-id>
# python play.py lpg_envs/TabularGridWorld-LongerHorizon-v0
```

---

## Environments Overview

All 15 registered environment IDs at a glance:

| Environment ID | Domain | Observation | Action | Max Steps |
|---|---|---|---|---|
| `lpg_envs/TabularGridWorld-Dense-v0` | Tabular Grid | `Discrete` (scalar) | `Discrete(9 or 18)` | 500 |
| `lpg_envs/TabularGridWorld-Sparse-v0` | Tabular Grid | `Discrete` (scalar) | `Discrete(9 or 18)` | 50 |
| `lpg_envs/TabularGridWorld-LongHorizon-v0` | Tabular Grid | `Discrete` (scalar) | `Discrete(9 or 18)` | 1000 |
| `lpg_envs/TabularGridWorld-LongerHorizon-v0` | Tabular Grid | `Discrete` (scalar) | `Discrete(9 or 18)` | 2000 |
| `lpg_envs/TabularGridWorld-LongDense-v0` | Tabular Grid | `Discrete` (scalar) | `Discrete(9 or 18)` | 2000 |
| `lpg_envs/RandomGridWorld-Dense-v0` | Random Grid | `MultiBinary(C,H,W)` | `Discrete(9 or 18)` | 500 |
| `lpg_envs/RandomGridWorld-LongHorizon-v0` | Random Grid | `MultiBinary(C,H,W)` | `Discrete(9 or 18)` | 1000 |
| `lpg_envs/RandomGridWorld-Small-v0` | Random Grid | `MultiBinary(C,H,W)` | `Discrete(9 or 18)` | 500 |
| `lpg_envs/RandomGridWorld-SmallSparse-v0` | Random Grid | `MultiBinary(C,H,W)` | `Discrete(9 or 18)` | 50 |
| `lpg_envs/RandomGridWorld-VeryDense-v0` | Random Grid | `MultiBinary(C,H,W)` | `Discrete(9 or 18)` | 2000 |
| `lpg_envs/DelayedChain-Short-v0` | Chain MDP | `Discrete` | `Discrete(2)` | chain length |
| `lpg_envs/DelayedChain-ShortNoisy-v0` | Chain MDP | `Discrete` | `Discrete(2)` | chain length |
| `lpg_envs/DelayedChain-Long-v0` | Chain MDP | `Discrete` | `Discrete(2)` | chain length |
| `lpg_envs/DelayedChain-LongNoisy-v0` | Chain MDP | `Discrete` | `Discrete(2)` | chain length |
| `lpg_envs/DelayedChain-Distractor-v0` | Chain MDP | `MultiBinary(22)` | `Discrete(2)` | chain length |


## Citation

This package implements the environments described in:

> Oh, J., Hessel, M., Czarnecki, W. M., Xu, Z., van Hasselt, H., Singh, S., & Silver, D. (2020).
> *Discovering Reinforcement Learning Algorithms.*
> arXiv preprint [arXiv:2007.08794](https://arxiv.org/abs/2007.08794).

---

## License

MIT
