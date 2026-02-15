"""Delayed Chain MDP environment from the LPG paper (Section A.3).

A binary-action MDP where the first action determines the reward received at
the end of the episode.  The chain length is sampled once per lifetime (per
environment instance) and stays fixed across episodes.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from lpg_envs.configs.chain_configs import CHAIN_CONFIGS, ChainMDPConfig


class DelayedChainMDPEnv(gymnasium.Env):
    """Delayed Chain MDP.

    Parameters
    ----------
    config : str | ChainMDPConfig
        Variant name (e.g. ``"short"``) looked up in ``CHAIN_CONFIGS``,
        or a ``ChainMDPConfig`` instance for custom setups.
    render_mode : str | None
        ``"ansi"`` for text rendering, or ``None``.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config: ChainMDPConfig | str = "short",
        render_mode: str | None = None,
    ):
        super().__init__()

        if isinstance(config, str):
            config = CHAIN_CONFIGS[config]

        self.config = config
        self.render_mode = render_mode

        # Chain length will be sampled on first reset (fixed per lifetime)
        self._chain_length: int | None = None

        # Observation space
        if config.distractor_bits > 0:
            obs_dim = 2 + config.distractor_bits  # 2 relevant + N noisy
            self.observation_space = gymnasium.spaces.MultiBinary(obs_dim)
        else:
            # +1 because the terminal observation equals chain_length
            self.observation_space = gymnasium.spaces.Discrete(
                config.chain_length_max + 1
            )

        self.action_space = gymnasium.spaces.Discrete(2)

        # Episode state
        self._current_step = 0
        self._correct_action: int = 0
        self._first_action: int | None = None
        self._noisy_rewards: np.ndarray | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Sample chain length once per lifetime
        if self._chain_length is None or seed is not None:
            self._chain_length = int(
                self.np_random.integers(
                    self.config.chain_length_min,
                    self.config.chain_length_max + 1,
                )
            )

        # Randomise correct action each episode
        self._correct_action = int(self.np_random.integers(0, 2))
        self._first_action = None
        self._current_step = 0

        # Pre-generate noisy rewards for intermediate states
        if self.config.noisy_rewards and self._chain_length > 2:
            self._noisy_rewards = self.np_random.choice(
                [-1.0, 1.0], size=self._chain_length - 2
            )
        else:
            self._noisy_rewards = None

        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action)

        # Record first action
        if self._current_step == 0:
            self._first_action = int(action)

        self._current_step += 1

        terminated = self._current_step >= self._chain_length
        reward = 0.0

        if terminated:
            reward = 1.0 if self._first_action == self._correct_action else -1.0
        elif self._noisy_rewards is not None and self._current_step >= 2:
            idx = self._current_step - 2
            if idx < len(self._noisy_rewards):
                reward = float(self._noisy_rewards[idx])

        truncated = False
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        if self.config.distractor_bits > 0:
            obs_dim = 2 + self.config.distractor_bits
            obs = np.zeros(obs_dim, dtype=np.int8)
            obs[0] = 1 if self._correct_action == 0 else 0
            if self._first_action is not None:
                obs[1] = (
                    1 if self._first_action == self._correct_action else 0
                )
            obs[2:] = self.np_random.integers(
                0, 2, size=self.config.distractor_bits
            )
            return obs
        return int(self._current_step)

    def _get_info(self) -> dict:
        return {
            "chain_length": self._chain_length,
            "current_step": self._current_step,
            "correct_action": self._correct_action,
        }

    def render(self):
        if self.render_mode == "ansi" and self._chain_length is not None:
            chain = ["."] * self._chain_length
            pos = min(self._current_step, self._chain_length - 1)
            chain[pos] = "A"
            return "Chain: [" + "".join(chain) + "]"
        return None

    def close(self):
        pass
