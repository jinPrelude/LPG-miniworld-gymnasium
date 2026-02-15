"""LPG training environments as Gymnasium custom environments.

Provides 15 environments across 3 domain types:
- 5 Tabular Grid World variants
- 5 Random Grid World variants
- 5 Delayed Chain MDP variants

Usage::

    import gymnasium
    import lpg_envs  # registers all environments

    env = gymnasium.make("lpg_envs/TabularGridWorld-Dense-v0")
    obs, info = env.reset()
"""

from gymnasium.envs.registration import register

# ============ Tabular Grid World (5 variants) ============

register(
    id="lpg_envs/TabularGridWorld-Dense-v0",
    entry_point="lpg_envs.envs:TabularGridWorldEnv",
    kwargs={"config": "dense"},
    max_episode_steps=500,
)
register(
    id="lpg_envs/TabularGridWorld-Sparse-v0",
    entry_point="lpg_envs.envs:TabularGridWorldEnv",
    kwargs={"config": "sparse"},
    max_episode_steps=50,
)
register(
    id="lpg_envs/TabularGridWorld-LongHorizon-v0",
    entry_point="lpg_envs.envs:TabularGridWorldEnv",
    kwargs={"config": "long_horizon"},
    max_episode_steps=1000,
)
register(
    id="lpg_envs/TabularGridWorld-LongerHorizon-v0",
    entry_point="lpg_envs.envs:TabularGridWorldEnv",
    kwargs={"config": "longer_horizon"},
    max_episode_steps=2000,
)
register(
    id="lpg_envs/TabularGridWorld-LongDense-v0",
    entry_point="lpg_envs.envs:TabularGridWorldEnv",
    kwargs={"config": "long_dense"},
    max_episode_steps=2000,
)

# ============ Random Grid World (5 variants) ============

register(
    id="lpg_envs/RandomGridWorld-Dense-v0",
    entry_point="lpg_envs.envs:RandomGridWorldEnv",
    kwargs={"config": "dense"},
    max_episode_steps=500,
)
register(
    id="lpg_envs/RandomGridWorld-LongHorizon-v0",
    entry_point="lpg_envs.envs:RandomGridWorldEnv",
    kwargs={"config": "long_horizon"},
    max_episode_steps=1000,
)
register(
    id="lpg_envs/RandomGridWorld-Small-v0",
    entry_point="lpg_envs.envs:RandomGridWorldEnv",
    kwargs={"config": "small"},
    max_episode_steps=500,
)
register(
    id="lpg_envs/RandomGridWorld-SmallSparse-v0",
    entry_point="lpg_envs.envs:RandomGridWorldEnv",
    kwargs={"config": "small_sparse"},
    max_episode_steps=50,
)
register(
    id="lpg_envs/RandomGridWorld-VeryDense-v0",
    entry_point="lpg_envs.envs:RandomGridWorldEnv",
    kwargs={"config": "very_dense"},
    max_episode_steps=2000,
)

# ============ Delayed Chain MDP (5 variants) ============

register(
    id="lpg_envs/DelayedChain-Short-v0",
    entry_point="lpg_envs.envs:DelayedChainMDPEnv",
    kwargs={"config": "short"},
)
register(
    id="lpg_envs/DelayedChain-ShortNoisy-v0",
    entry_point="lpg_envs.envs:DelayedChainMDPEnv",
    kwargs={"config": "short_noisy"},
)
register(
    id="lpg_envs/DelayedChain-Long-v0",
    entry_point="lpg_envs.envs:DelayedChainMDPEnv",
    kwargs={"config": "long"},
)
register(
    id="lpg_envs/DelayedChain-LongNoisy-v0",
    entry_point="lpg_envs.envs:DelayedChainMDPEnv",
    kwargs={"config": "long_noisy"},
)
register(
    id="lpg_envs/DelayedChain-Distractor-v0",
    entry_point="lpg_envs.envs:DelayedChainMDPEnv",
    kwargs={"config": "distractor"},
)
