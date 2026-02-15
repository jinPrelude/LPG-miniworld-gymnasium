"""Shared test fixtures."""

import lpg_envs  # noqa: F401 — triggers environment registration

ALL_ENV_IDS = [
    # Tabular Grid World
    "lpg_envs/TabularGridWorld-Dense-v0",
    "lpg_envs/TabularGridWorld-Sparse-v0",
    "lpg_envs/TabularGridWorld-LongHorizon-v0",
    "lpg_envs/TabularGridWorld-LongerHorizon-v0",
    "lpg_envs/TabularGridWorld-LongDense-v0",
    # Random Grid World
    "lpg_envs/RandomGridWorld-Dense-v0",
    "lpg_envs/RandomGridWorld-LongHorizon-v0",
    "lpg_envs/RandomGridWorld-Small-v0",
    "lpg_envs/RandomGridWorld-SmallSparse-v0",
    "lpg_envs/RandomGridWorld-VeryDense-v0",
    # Delayed Chain MDP
    "lpg_envs/DelayedChain-Short-v0",
    "lpg_envs/DelayedChain-ShortNoisy-v0",
    "lpg_envs/DelayedChain-Long-v0",
    "lpg_envs/DelayedChain-LongNoisy-v0",
    "lpg_envs/DelayedChain-Distractor-v0",
]
