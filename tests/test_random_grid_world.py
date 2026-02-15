"""Tests for RandomGridWorldEnv."""

import numpy as np
import pytest

from lpg_envs.configs.grid_world_configs import RANDOM_CONFIGS
from lpg_envs.envs.random_grid_world import RandomGridWorldEnv


class TestObservation:
    """Verify observation shape and content."""

    @pytest.mark.parametrize("key", RANDOM_CONFIGS.keys())
    def test_obs_shape(self, key):
        cfg = RANDOM_CONFIGS[key]
        env = RandomGridWorldEnv(config=cfg, action_mode="move_only")
        obs, _ = env.reset(seed=0)
        expected_channels = len(cfg.objects) + 1
        assert obs.shape == (expected_channels, env.height, env.width)
        env.close()

    def test_obs_binary(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        assert set(np.unique(obs)).issubset({0, 1})
        env.close()

    def test_agent_channel(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        # Channel 0 should have exactly one 1 (the agent)
        assert obs[0].sum() == 1
        r, c = env._agent_pos
        assert obs[0, r, c] == 1
        env.close()

    def test_object_channels(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        for i in range(env.num_objects):
            if env._object_present[i]:
                assert obs[i + 1].sum() == 1
                r, c = env._object_positions[i]
                assert obs[i + 1, r, c] == 1
        env.close()

    def test_obs_in_space(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(20):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        env.close()


class TestPerEpisodeRandomisation:
    """Object positions should change between episodes."""

    def test_positions_differ_across_resets(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=0)
        pos_first = list(env._object_positions)

        env.reset()
        pos_second = list(env._object_positions)

        # It's extremely unlikely that all positions are identical
        assert pos_first != pos_second
        env.close()


class TestRespawn:
    """Respawned objects should appear at random positions."""

    def test_respawn_at_random(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=0)
        original_pos = env._object_positions[0]

        # Collect and respawn many times — at least one should differ
        positions_seen = {original_pos}
        env._object_present[0] = False
        for _ in range(50):
            env._respawn_object(0)
            positions_seen.add(env._object_positions[0])

        assert len(positions_seen) > 1
        env.close()
