"""Tests for TabularGridWorldEnv."""

import numpy as np
import pytest

from lpg_envs.configs.grid_world_configs import TABULAR_CONFIGS
from lpg_envs.envs.tabular_grid_world import TabularGridWorldEnv


class TestStateSpace:
    """Verify observation space matches p * 2^m."""

    @pytest.mark.parametrize("key", TABULAR_CONFIGS.keys())
    def test_state_count(self, key):
        cfg = TABULAR_CONFIGS[key]
        env = TabularGridWorldEnv(config=cfg, action_mode="move_only")
        expected = env.height * env.width * (2 ** len(cfg.objects))
        assert env.observation_space.n == expected
        env.close()


class TestLifetimeSemantics:
    """Object positions should be fixed within a lifetime."""

    def test_positions_fixed_across_resets(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=42)
        pos_first = list(env._fixed_positions)

        # Reset again (same lifetime) — positions should stay
        env.reset()
        assert env._fixed_positions == pos_first

        # Reset with same seed — positions should stay
        env.reset(seed=42)
        assert env._fixed_positions == pos_first
        env.close()

    def test_positions_change_with_new_seed(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=42)
        pos_a = list(env._fixed_positions)

        env.reset(seed=999)
        pos_b = list(env._fixed_positions)

        # Very unlikely to be identical with different seeds
        assert pos_a != pos_b
        env.close()


class TestObservation:
    """Verify observation encoding."""

    def test_obs_in_space(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_only")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(20):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
        env.close()

    def test_obs_encodes_position_and_objects(self):
        env = TabularGridWorldEnv(config="sparse", action_mode="move_only")
        env.reset(seed=0)
        num_objects = env.num_objects
        obs = env._get_obs()

        # Decode
        bitmask = obs % (2 ** num_objects)
        agent_idx = obs // (2 ** num_objects)
        row, col = divmod(agent_idx, env.width)

        assert (row, col) == env._agent_pos
        for i in range(num_objects):
            expected = env._object_present[i]
            assert bool(bitmask & (1 << i)) == expected
        env.close()


class TestActionModes:
    """Test both 9-action and 18-action modes."""

    def test_move_only(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_only")
        assert env.action_space.n == 9
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
        env.close()

    def test_move_and_collect(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_and_collect")
        assert env.action_space.n == 18
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
        env.close()


class TestCollection:
    """Test object collection mechanics."""

    def test_collect_gives_reward(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=0)

        # Manually place agent on an object
        env._agent_pos = env._object_positions[0]
        reward, terminated = env._try_collect_at_current()
        assert reward == env.config.objects[0].reward
        assert not env._object_present[0]
        env.close()

    def test_respawn_at_fixed_position(self):
        env = TabularGridWorldEnv(config="dense", action_mode="move_only")
        env.reset(seed=0)
        original_pos = env._object_positions[0]

        # Collect object 0
        env._object_present[0] = False

        # Respawn it
        env._respawn_object(0)
        env._object_present[0] = True

        assert env._object_positions[0] == original_pos
        env.close()
