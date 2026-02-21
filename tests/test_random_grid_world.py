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
        num_channels = len(cfg.objects) + 1
        expected_dim = num_channels * env.height * env.width
        assert obs.shape == (expected_dim,)
        env.close()

    def test_obs_is_float32(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        env.close()

    def test_obs_binary(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        assert set(np.unique(obs)).issubset({0.0, 1.0})
        env.close()

    def test_agent_channel(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        num_cells = env.height * env.width
        agent_channel = obs[:num_cells]
        # Exactly one 1 in agent channel
        assert agent_channel.sum() == 1.0
        r, c = env._agent_pos
        assert agent_channel[r * env.width + c] == 1.0
        env.close()

    def test_object_channels(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        num_cells = env.height * env.width
        for i in range(env.num_objects):
            obj_channel = obs[(i + 1) * num_cells : (i + 2) * num_cells]
            if env._object_present[i]:
                assert obj_channel.sum() == 1.0
                r, c = env._object_positions[i]
                assert obj_channel[r * env.width + c] == 1.0
            else:
                assert obj_channel.sum() == 0.0
        env.close()

    def test_obs_in_space(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(20):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        env.close()

    def test_obs_all_binary_after_step(self):
        env = RandomGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(0)
        assert set(np.unique(obs)).issubset({0.0, 1.0})
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
