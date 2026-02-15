"""Load wall maps from text files.

A map file uses '#' for walls and ' ' (space) for walkable floor.
Lines are padded to the width of the longest line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Default maps directory (project_root/maps/)
_MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "maps"


def load_wall_map(map_name: str, maps_dir: Path | str | None = None) -> np.ndarray:
    """Load a wall map from a text file.

    Parameters
    ----------
    map_name : str
        Name of the map (without .txt extension).
    maps_dir : Path | str | None
        Directory containing map files.  Defaults to ``<project_root>/maps/``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(height, width)`` where ``True`` = wall.
    """
    if maps_dir is None:
        maps_dir = _MAPS_DIR
    maps_dir = Path(maps_dir)

    filepath = maps_dir / f"{map_name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Map file not found: {filepath}")

    with open(filepath) as f:
        raw_lines = f.readlines()

    # Strip trailing newlines but keep content (including spaces)
    lines = [line.rstrip("\n\r") for line in raw_lines]

    # Remove empty leading/trailing lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        raise ValueError(f"Map file is empty: {filepath}")

    # Pad all lines to the same width
    max_width = max(len(line) for line in lines)
    lines = [line.ljust(max_width) for line in lines]

    height = len(lines)
    width = max_width

    wall_map = np.zeros((height, width), dtype=bool)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "#":
                wall_map[r, c] = True

    return wall_map
