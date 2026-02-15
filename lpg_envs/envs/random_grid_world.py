"""Random Grid World environment from the LPG paper (Section A.2).

Object locations are randomised every episode.  Objects that are collected
re-appear at random locations.  Observation is a binary tensor
``{0,1}^{(num_objects+1) x H x W}`` with channel 0 for the agent and one
channel per object instance.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from lpg_envs.configs.grid_world_configs import RANDOM_CONFIGS, GridWorldConfig
from lpg_envs.envs.grid_world_base import GridWorldBase


class RandomGridWorldEnv(GridWorldBase):
    """Random Grid World with per-episode randomised object placement.

    Parameters
    ----------
    config : str | GridWorldConfig
        Variant name (e.g. ``"dense"``) looked up in ``RANDOM_CONFIGS``,
        or a ``GridWorldConfig`` instance for custom setups.
    action_mode : str
        ``"auto"`` (random 9/18), ``"move_only"`` (9), or
        ``"move_and_collect"`` (18).
    render_mode : str | None
        ``"human"``, ``"rgb_array"``, or ``None``.
    """

    def __init__(
        self,
        config: GridWorldConfig | str = "dense",
        action_mode: str = "auto",
        render_mode: str | None = None,
    ):
        super().__init__(
            config=config,
            config_registry=RANDOM_CONFIGS,
            action_mode=action_mode,
            render_mode=render_mode,
        )

    def _define_observation_space(self) -> gymnasium.spaces.Space:
        num_channels = self.num_objects + 1  # +1 for agent channel
        return gymnasium.spaces.MultiBinary(
            [num_channels, self.height, self.width]
        )

    def _place_objects(self) -> None:
        """Place objects at random non-overlapping floor positions (every episode)."""
        occupied: set[tuple[int, int]] = set()
        self._object_positions = []
        for _ in range(self.num_objects):
            pos = self._random_floor_position(occupied)
            self._object_positions.append(pos)
            occupied.add(pos)

    def _respawn_object(self, obj_idx: int) -> None:
        """Respawn at a random unoccupied floor position."""
        occupied: set[tuple[int, int]] = {self._agent_pos}
        for i in range(self.num_objects):
            if self._object_present[i] and i != obj_idx:
                occupied.add(self._object_positions[i])
        self._object_positions[obj_idx] = self._random_floor_position(occupied)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(
            (self.num_objects + 1, self.height, self.width), dtype=np.int8
        )
        obs[0, self._agent_pos[0], self._agent_pos[1]] = 1
        for i in range(self.num_objects):
            if self._object_present[i]:
                r, c = self._object_positions[i]
                obs[i + 1, r, c] = 1
        return obs
