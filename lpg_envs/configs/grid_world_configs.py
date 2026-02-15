"""Grid world environment configurations from the LPG paper."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectDef:
    """Definition of a single object in the grid world.

    Attributes:
        reward: Reward received when the object is collected.
        term_prob: Probability that the episode terminates upon collection.
        respawn_prob: Per-timestep probability of reappearing after collection.
    """

    reward: float
    term_prob: float
    respawn_prob: float


@dataclass(frozen=True)
class GridWorldConfig:
    """Full configuration for a grid world variant.

    Attributes:
        name: Human-readable variant name.
        grid_height: Number of rows in the grid.
        grid_width: Number of columns in the grid.
        objects: Tuple of ObjectDef instances (one per object instance).
        max_episode_steps: Maximum steps before episode is truncated.
        map_name: Optional name of a map file in the maps/ directory.
            When set, the wall layout is loaded from that file and
            grid_height/grid_width are overridden by the map dimensions.
    """

    name: str
    grid_height: int
    grid_width: int
    objects: tuple[ObjectDef, ...]
    max_episode_steps: int
    map_name: str | None = None


# ---------------------------------------------------------------------------
# Tabular Grid World variants (Section A.1)
# Object locations fixed per lifetime, observation = state index (integer)
# ---------------------------------------------------------------------------

TABULAR_CONFIGS: dict[str, GridWorldConfig] = {
    "dense": GridWorldConfig(
        name="Dense",
        grid_height=11,
        grid_width=11,
        objects=(
            ObjectDef(1.0, 0.0, 0.05),
            ObjectDef(1.0, 0.0, 0.05),
            ObjectDef(-1.0, 0.5, 0.1),
            ObjectDef(-1.0, 0.0, 0.5),
        ),
        max_episode_steps=500,
        map_name="tabular_dense",
    ),
    "sparse": GridWorldConfig(
        name="Sparse",
        grid_height=13,
        grid_width=13,
        objects=(
            ObjectDef(1.0, 1.0, 0.0),
            ObjectDef(-1.0, 1.0, 0.0),
        ),
        max_episode_steps=50,
        map_name="tabular_sparse",
    ),
    "long_horizon": GridWorldConfig(
        name="LongHorizon",
        grid_height=11,
        grid_width=11,
        objects=(
            ObjectDef(1.0, 0.0, 0.01),
            ObjectDef(1.0, 0.0, 0.01),
            ObjectDef(-1.0, 0.5, 1.0),
            ObjectDef(-1.0, 0.5, 1.0),
        ),
        max_episode_steps=1000,
        map_name="tabular_long_horizon",
    ),
    "longer_horizon": GridWorldConfig(
        name="LongerHorizon",
        grid_height=7,
        grid_width=9,
        objects=(
            ObjectDef(1.0, 0.1, 0.01),
            ObjectDef(1.0, 0.1, 0.01),
            ObjectDef(-1.0, 0.8, 1.0),
            ObjectDef(-1.0, 0.8, 1.0),
            ObjectDef(-1.0, 0.8, 1.0),
            ObjectDef(-1.0, 0.8, 1.0),
            ObjectDef(-1.0, 0.8, 1.0),
        ),
        max_episode_steps=2000,
        map_name="tabular_longer_horizon",
    ),
    "long_dense": GridWorldConfig(
        name="LongDense",
        grid_height=11,
        grid_width=11,
        objects=(
            ObjectDef(1.0, 0.0, 0.005),
            ObjectDef(1.0, 0.0, 0.005),
            ObjectDef(1.0, 0.0, 0.005),
            ObjectDef(1.0, 0.0, 0.005),
        ),
        max_episode_steps=2000,
        map_name="tabular_long_dense",
    ),
}

# ---------------------------------------------------------------------------
# Random Grid World variants (Section A.2)
# Object locations randomized every episode, observation = binary tensor
# ---------------------------------------------------------------------------

RANDOM_CONFIGS: dict[str, GridWorldConfig] = {
    "dense": GridWorldConfig(
        name="Dense",
        grid_height=11,
        grid_width=11,
        objects=(
            ObjectDef(1.0, 0.0, 0.05),
            ObjectDef(1.0, 0.0, 0.05),
            ObjectDef(-1.0, 0.5, 0.1),
            ObjectDef(-1.0, 0.0, 0.5),
        ),
        max_episode_steps=500,
        map_name="random_dense",
    ),
    "long_horizon": GridWorldConfig(
        name="LongHorizon",
        grid_height=11,
        grid_width=11,
        objects=(
            ObjectDef(1.0, 0.0, 0.01),
            ObjectDef(1.0, 0.0, 0.01),
            ObjectDef(-1.0, 0.5, 1.0),
            ObjectDef(-1.0, 0.5, 1.0),
        ),
        max_episode_steps=1000,
        map_name="random_long_horizon",
    ),
    "small": GridWorldConfig(
        name="Small",
        grid_height=5,
        grid_width=7,
        objects=(
            ObjectDef(1.0, 0.0, 0.05),
            ObjectDef(1.0, 0.0, 0.05),
            ObjectDef(-1.0, 0.5, 0.1),
            ObjectDef(-1.0, 0.5, 0.1),
        ),
        max_episode_steps=500,
        map_name="random_small",
    ),
    "small_sparse": GridWorldConfig(
        name="SmallSparse",
        grid_height=5,
        grid_width=7,
        objects=(
            ObjectDef(1.0, 1.0, 1.0),
            ObjectDef(-1.0, 1.0, 1.0),
            ObjectDef(-1.0, 1.0, 1.0),
        ),
        max_episode_steps=50,
        map_name="random_small_sparse",
    ),
    "very_dense": GridWorldConfig(
        name="VeryDense",
        grid_height=11,
        grid_width=11,
        objects=(ObjectDef(1.0, 0.0, 1.0),),
        max_episode_steps=2000,
        map_name="random_very_dense",
    ),
}
