"""Tabular Grid World environment from the LPG paper (Section A.1).

Object locations are fixed per lifetime (i.e., per environment instance) and
randomised across lifetimes via the seed.  Observation is a one-hot float32
vector encoding the (position, object-presence) state.
"""

from __future__ import annotations

import gymnasium
import numpy as np

from lpg_envs.configs.grid_world_configs import TABULAR_CONFIGS, GridWorldConfig
from lpg_envs.envs.grid_world_base import GridWorldBase


class TabularGridWorldEnv(GridWorldBase):
    """Tabular Grid World with fixed object positions per lifetime.

    Parameters
    ----------
    config : str | GridWorldConfig
        Variant name (e.g. ``"dense"``) looked up in ``TABULAR_CONFIGS``,
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
        self._fixed_positions: list[tuple[int, int]] | None = None
        super().__init__(
            config=config,
            config_registry=TABULAR_CONFIGS,
            action_mode=action_mode,
            render_mode=render_mode,
        )

    def _define_observation_space(self) -> gymnasium.spaces.Space:
        num_positions = self.height * self.width
        self._num_tabular_states = num_positions * (2 ** self.num_objects)
        return gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._num_tabular_states,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        # When a new seed is provided (= new lifetime) or first call,
        # invalidate fixed positions so _place_objects() re-generates them.
        if seed is not None or self._fixed_positions is None:
            self._fixed_positions = None

        return GridWorldBase.reset(self, seed=seed, options=options)

    def _place_objects_fixed(self) -> None:
        """Generate random non-overlapping positions, fixed for the lifetime."""
        occupied: set[tuple[int, int]] = set()
        self._fixed_positions = []
        for _ in range(self.num_objects):
            pos = self._random_floor_position(occupied)
            occupied.add(pos)
            self._fixed_positions.append(pos)

    def _place_objects(self) -> None:
        """Restore objects to their fixed lifetime positions.

        On first call (or after a new seed), generates the fixed positions.
        """
        if self._fixed_positions is None:
            self._place_objects_fixed()
        self._object_positions = list(self._fixed_positions)

    def _respawn_object(self, obj_idx: int) -> None:
        """Respawn at original fixed position."""
        assert self._fixed_positions is not None
        self._object_positions[obj_idx] = self._fixed_positions[obj_idx]

    def _get_obs(self) -> np.ndarray:
        agent_idx = self._agent_pos[0] * self.width + self._agent_pos[1]
        bitmask = 0
        for i in range(self.num_objects):
            if self._object_present[i]:
                bitmask |= 1 << i
        state_idx = agent_idx * (2 ** self.num_objects) + bitmask
        obs = np.zeros(self._num_tabular_states, dtype=np.float32)
        obs[state_idx] = 1.0
        return obs
