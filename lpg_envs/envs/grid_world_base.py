"""Abstract base class for LPG grid world environments."""

from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium
import numpy as np
import pygame

from lpg_envs.configs.grid_world_configs import GridWorldConfig
from lpg_envs.configs.map_loader import load_wall_map


# 9 directions: stay + 8 adjacent (row_delta, col_delta)
_DIRECTIONS = [
    (0, 0),    # 0: stay
    (-1, 0),   # 1: N
    (-1, 1),   # 2: NE
    (0, 1),    # 3: E
    (1, 1),    # 4: SE
    (1, 0),    # 5: S
    (1, -1),   # 6: SW
    (0, -1),   # 7: W
    (-1, -1),  # 8: NW
]


class GridWorldBase(gymnasium.Env, ABC):
    """Abstract base for Tabular and Random grid world environments.

    Subclasses must implement:
        _define_observation_space() -> gymnasium.spaces.Space
        _get_obs() -> observation
        _place_objects() -> None  (set self._object_positions)
        _respawn_object(obj_idx) -> None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: GridWorldConfig | str,
        config_registry: dict[str, GridWorldConfig] | None = None,
        action_mode: str = "auto",
        render_mode: str | None = None,
    ):
        super().__init__()

        if isinstance(config, str):
            assert config_registry is not None
            config = config_registry[config]

        self.config = config
        self.render_mode = render_mode

        # Load wall map if specified; dimensions come from the map
        if config.map_name is not None:
            self._wall_map = load_wall_map(config.map_name)
            self.height, self.width = self._wall_map.shape
        else:
            self.height = config.grid_height
            self.width = config.grid_width
            self._wall_map = np.zeros((self.height, self.width), dtype=bool)

        self.num_objects = len(config.objects)

        # Action mode
        if action_mode == "auto":
            # Sample randomly; use Python random since np_random isn't seeded yet
            import random
            self._num_actions = random.choice([9, 18])
        elif action_mode == "move_only":
            self._num_actions = 9
        elif action_mode == "move_and_collect":
            self._num_actions = 18
        else:
            self._num_actions = int(action_mode)

        self.action_space = gymnasium.spaces.Discrete(self._num_actions)
        self.observation_space = self._define_observation_space()

        # Internal state (initialised in reset)
        self._agent_pos: tuple[int, int] = (0, 0)
        self._object_positions: list[tuple[int, int]] = [(0, 0)] * self.num_objects
        self._object_present = np.ones(self.num_objects, dtype=bool)
        self._step_count = 0

        # Rendering
        self._window = None
        self._clock = None
        self._cell_size = 48

    def _is_wall(self, r: int, c: int) -> bool:
        """Return True if the cell at (r, c) is a wall or out of bounds."""
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return bool(self._wall_map[r, c])

    def _random_floor_position(
        self, occupied: set[tuple[int, int]] | None = None
    ) -> tuple[int, int]:
        """Return a random non-wall position that is not in *occupied*."""
        if occupied is None:
            occupied = set()
        while True:
            pos = (
                int(self.np_random.integers(0, self.height)),
                int(self.np_random.integers(0, self.width)),
            )
            if not self._is_wall(pos[0], pos[1]) and pos not in occupied:
                return pos

    @abstractmethod
    def _define_observation_space(self) -> gymnasium.spaces.Space:
        """Return the observation space for this environment type."""

    @abstractmethod
    def _get_obs(self):
        """Return the current observation."""

    @abstractmethod
    def _place_objects(self) -> None:
        """Place objects on the grid (called from reset)."""

    @abstractmethod
    def _respawn_object(self, obj_idx: int) -> None:
        """Respawn a collected object."""

    def _get_info(self) -> dict:
        return {
            "agent_pos": self._agent_pos,
            "object_present": self._object_present.copy(),
            "step_count": self._step_count,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._place_objects()

        # Random agent start position (must be on floor)
        self._agent_pos = self._random_floor_position()

        self._object_present = np.ones(self.num_objects, dtype=bool)
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if self._num_actions == 18:
            if action < 9:
                self._move_agent(action)
            else:
                # Collect at adjacent position (direction = action - 9)
                reward, terminated = self._try_collect_at(action - 9)
        else:
            # 9-action mode: move then auto-collect at current position
            self._move_agent(action)
            reward, terminated = self._try_collect_at_current()

        # Respawn absent objects
        for i in range(self.num_objects):
            if not self._object_present[i]:
                if self.np_random.random() < self.config.objects[i].respawn_prob:
                    self._respawn_object(i)
                    self._object_present[i] = True

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _move_agent(self, direction_idx: int) -> None:
        dr, dc = _DIRECTIONS[direction_idx]
        new_r = self._agent_pos[0] + dr
        new_c = self._agent_pos[1] + dc
        # Only move if the target cell is not a wall
        if not self._is_wall(new_r, new_c):
            self._agent_pos = (new_r, new_c)

    def _try_collect_at_current(self) -> tuple[float, bool]:
        """Collect any object at the agent's current position."""
        for i in range(self.num_objects):
            if self._object_present[i] and self._object_positions[i] == self._agent_pos:
                return self._collect_object(i)
        return 0.0, False

    def _try_collect_at(self, direction_idx: int) -> tuple[float, bool]:
        """Collect any object at an adjacent position."""
        dr, dc = _DIRECTIONS[direction_idx]
        target_r = self._agent_pos[0] + dr
        target_c = self._agent_pos[1] + dc
        # Can't collect through walls or out of bounds
        if self._is_wall(target_r, target_c):
            return 0.0, False
        target_pos = (target_r, target_c)

        for i in range(self.num_objects):
            if self._object_present[i] and self._object_positions[i] == target_pos:
                return self._collect_object(i)
        return 0.0, False

    def _collect_object(self, obj_idx: int) -> tuple[float, bool]:
        """Collect a specific object. Returns (reward, terminated)."""
        obj = self.config.objects[obj_idx]
        self._object_present[obj_idx] = False
        reward = obj.reward
        terminated = bool(self.np_random.random() < obj.term_prob)
        return reward, terminated

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray | None:
        canvas_w = self.width * self._cell_size
        canvas_h = self.height * self._cell_size

        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((canvas_w, canvas_h))
        if self._clock is None and self.render_mode == "human":
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((canvas_w, canvas_h))
        canvas.fill((255, 255, 255))

        cs = self._cell_size

        # Draw walls
        for r in range(self.height):
            for c in range(self.width):
                if self._wall_map[r, c]:
                    wall_rect = pygame.Rect(c * cs, r * cs, cs, cs)
                    pygame.draw.rect(canvas, (60, 60, 60), wall_rect)

        # Draw grid lines
        for r in range(self.height + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, r * cs), (canvas_w, r * cs))
        for c in range(self.width + 1):
            pygame.draw.line(canvas, (200, 200, 200), (c * cs, 0), (c * cs, canvas_h))

        # Draw objects
        for i in range(self.num_objects):
            if self._object_present[i]:
                r, c = self._object_positions[i]
                obj = self.config.objects[i]
                if obj.reward > 0:
                    color = (255, 215, 0)  # gold for positive
                else:
                    color = (220, 50, 50)  # red for negative
                center = (c * cs + cs // 2, r * cs + cs // 2)
                pygame.draw.circle(canvas, color, center, cs // 3)

        # Draw agent
        ar, ac = self._agent_pos
        agent_rect = pygame.Rect(ac * cs + cs // 4, ar * cs + cs // 4, cs // 2, cs // 2)
        pygame.draw.rect(canvas, (0, 100, 255), agent_rect)

        if self.render_mode == "human":
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
