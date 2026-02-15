"""Test that all 15 environments can be created via gymnasium.make()."""

import gymnasium
import pytest

from tests.conftest import ALL_ENV_IDS


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_make_and_reset(env_id):
    env = gymnasium.make(env_id)
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert isinstance(info, dict)
    env.close()


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_step(env_id):
    env = gymnasium.make(env_id)
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()
