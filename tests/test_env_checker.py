"""Run gymnasium env_checker on all environments."""

import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env

from tests.conftest import ALL_ENV_IDS


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_env_checker(env_id):
    env = gymnasium.make(env_id, disable_env_checker=True).unwrapped
    check_env(env)
    env.close()
