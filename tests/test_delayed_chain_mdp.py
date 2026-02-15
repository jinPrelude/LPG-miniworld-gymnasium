"""Tests for DelayedChainMDPEnv."""

import numpy as np
import pytest

from lpg_envs.configs.chain_configs import CHAIN_CONFIGS
from lpg_envs.envs.delayed_chain_mdp import DelayedChainMDPEnv


class TestDelayedReward:
    """The reward should only come at the last step."""

    def test_correct_action_gives_positive_reward(self):
        env = DelayedChainMDPEnv(config="short")
        env.reset(seed=42)
        correct = env._correct_action
        chain_len = env._chain_length

        # First step: choose correct action
        obs, r, term, trunc, info = env.step(correct)
        assert r == 0.0  # No reward at step 1 (unless chain_len == 1)

        # Step until end
        for _ in range(chain_len - 2):
            obs, r, term, trunc, info = env.step(0)
            if not env.config.noisy_rewards:
                assert r == 0.0

        # Last step should give +1
        obs, r, term, trunc, info = env.step(0)
        assert r == 1.0
        assert term is True
        env.close()

    def test_wrong_action_gives_negative_reward(self):
        env = DelayedChainMDPEnv(config="short")
        env.reset(seed=42)
        wrong = 1 - env._correct_action
        chain_len = env._chain_length

        # First step: choose wrong action
        env.step(wrong)
        for _ in range(chain_len - 2):
            env.step(0)
        obs, r, term, trunc, info = env.step(0)
        assert r == -1.0
        assert term is True
        env.close()


class TestChainLength:
    """Chain length should be fixed within a lifetime."""

    def test_fixed_within_lifetime(self):
        env = DelayedChainMDPEnv(config="long")
        env.reset(seed=42)
        cl = env._chain_length

        for _ in range(10):
            env.reset()
            assert env._chain_length == cl
        env.close()

    def test_changes_with_new_seed(self):
        env = DelayedChainMDPEnv(config="long")
        lengths = set()
        for seed in range(50):
            env.reset(seed=seed)
            lengths.add(env._chain_length)
        # Should see multiple different chain lengths
        assert len(lengths) > 1
        env.close()

    @pytest.mark.parametrize("key", CHAIN_CONFIGS.keys())
    def test_chain_length_in_range(self, key):
        cfg = CHAIN_CONFIGS[key]
        env = DelayedChainMDPEnv(config=cfg)
        for seed in range(20):
            env.reset(seed=seed)
            assert cfg.chain_length_min <= env._chain_length <= cfg.chain_length_max
        env.close()


class TestCorrectActionRandomisation:
    """Correct action should vary across episodes."""

    def test_correct_action_varies(self):
        env = DelayedChainMDPEnv(config="short")
        env.reset(seed=42)
        actions_seen = set()
        for _ in range(20):
            env.reset()
            actions_seen.add(env._correct_action)
        assert actions_seen == {0, 1}
        env.close()


class TestNoisyRewards:
    """Test noisy reward variants."""

    def test_noisy_has_intermediate_rewards(self):
        env = DelayedChainMDPEnv(config="short_noisy")
        env.reset(seed=42)
        chain_len = env._chain_length

        # First action
        env.step(env._correct_action)

        # Check intermediate rewards
        intermediate_rewards = []
        for _ in range(chain_len - 2):
            _, r, _, _, _ = env.step(0)
            intermediate_rewards.append(r)

        # At least some non-zero intermediate rewards
        if chain_len > 2:
            assert any(r != 0.0 for r in intermediate_rewards)
            assert all(r in (-1.0, 1.0) for r in intermediate_rewards)
        env.close()

    def test_non_noisy_has_no_intermediate_rewards(self):
        env = DelayedChainMDPEnv(config="short")
        env.reset(seed=42)
        chain_len = env._chain_length

        env.step(env._correct_action)
        for _ in range(chain_len - 2):
            _, r, _, _, _ = env.step(0)
            assert r == 0.0
        env.close()


class TestDistractor:
    """Test distractor variant with binary observations."""

    def test_obs_shape(self):
        env = DelayedChainMDPEnv(config="distractor")
        obs, _ = env.reset(seed=0)
        assert obs.shape == (22,)
        assert set(np.unique(obs)).issubset({0, 1})
        env.close()

    def test_obs_in_space(self):
        env = DelayedChainMDPEnv(config="distractor")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(5):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        env.close()

    def test_relevant_bits(self):
        env = DelayedChainMDPEnv(config="distractor")
        env.reset(seed=42)
        obs = env._get_obs()
        # Bit 0: whether a0 is correct
        if env._correct_action == 0:
            assert obs[0] == 1
        else:
            assert obs[0] == 0
        env.close()


class TestObservationSpace:
    """Non-distractor variants should return integer observations."""

    @pytest.mark.parametrize("key", ["short", "short_noisy", "long", "long_noisy"])
    def test_discrete_obs(self, key):
        env = DelayedChainMDPEnv(config=key)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, int)
        assert env.observation_space.contains(obs)
        env.close()
