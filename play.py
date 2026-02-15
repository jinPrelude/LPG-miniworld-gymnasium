#!/usr/bin/env python3
"""Interactive play script for all LPG environments using pygame.

Usage:
    python play.py                    # opens environment selection menu
    python play.py <env_id>           # play a specific environment directly
    python play.py --list             # list all available environment IDs

Grid World Controls:
    Arrow keys / WASD   Cardinal movement (N/S/E/W)
    Q / E / Z / C       Diagonal movement (NW/NE/SW/SE)
    Space               Stay in place

Chain MDP Controls:
    Left arrow / 0      Action 0
    Right arrow / 1     Action 1

Common Controls:
    R                   Reset episode
    N                   New lifetime (reset with new seed)
    ESC                 Back to menu / quit
"""

from __future__ import annotations

import argparse
import random
import sys

import gymnasium
import numpy as np
import pygame

import lpg_envs  # noqa: F401 – registers environments

# ── Colours ──────────────────────────────────────────────────────────────────

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GREY = (60, 60, 60)
BORDER_INNER = (50, 50, 50)
LIGHT_GREY = (200, 200, 200)
MID_GREY = (140, 140, 140)
BLUE = (0, 100, 255)
GOLD = (255, 215, 0)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
BG_COLOR = (30, 30, 30)
HUD_BG = (40, 40, 40)
HUD_TEXT = (220, 220, 220)
MENU_HIGHLIGHT = (70, 130, 200)
MENU_BG = (25, 25, 35)

# ── Environment registry ────────────────────────────────────────────────────

ENV_GROUPS = {
    "Tabular Grid World": [
        "lpg_envs/TabularGridWorld-Dense-v0",
        "lpg_envs/TabularGridWorld-Sparse-v0",
        "lpg_envs/TabularGridWorld-LongHorizon-v0",
        "lpg_envs/TabularGridWorld-LongerHorizon-v0",
        "lpg_envs/TabularGridWorld-LongDense-v0",
    ],
    "Random Grid World": [
        "lpg_envs/RandomGridWorld-Dense-v0",
        "lpg_envs/RandomGridWorld-LongHorizon-v0",
        "lpg_envs/RandomGridWorld-Small-v0",
        "lpg_envs/RandomGridWorld-SmallSparse-v0",
        "lpg_envs/RandomGridWorld-VeryDense-v0",
    ],
    "Delayed Chain MDP": [
        "lpg_envs/DelayedChain-Short-v0",
        "lpg_envs/DelayedChain-ShortNoisy-v0",
        "lpg_envs/DelayedChain-Long-v0",
        "lpg_envs/DelayedChain-LongNoisy-v0",
        "lpg_envs/DelayedChain-Distractor-v0",
    ],
}

ALL_ENV_IDS: list[str] = []
for _ids in ENV_GROUPS.values():
    ALL_ENV_IDS.extend(_ids)

# ── Layout constants ────────────────────────────────────────────────────────

CELL_SIZE = 48
HUD_HEIGHT = 60
CHAIN_WIDTH = 800
CHAIN_HEIGHT = 280
FPS = 30


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_grid_world(env_id: str) -> bool:
    return "GridWorld" in env_id


# ── Grid‑world key → action mapping ─────────────────────────────────────────
#
#   Q  W  E        NW  N  NE
#   A     D   →    W   ·  E
#   Z  S  C        SW  S  SE
#
#   Arrow keys duplicate WASD for cardinal directions.
#   Space = stay.

_GRID_KEY_MAP: dict[int, int] = {
    pygame.K_SPACE: 0,   # stay
    pygame.K_UP: 1,      # N
    pygame.K_w: 1,
    pygame.K_e: 2,       # NE
    pygame.K_RIGHT: 3,   # E
    pygame.K_d: 3,
    pygame.K_c: 4,       # SE
    pygame.K_DOWN: 5,    # S
    pygame.K_s: 5,
    pygame.K_z: 6,       # SW
    pygame.K_LEFT: 7,    # W
    pygame.K_a: 7,
    pygame.K_q: 8,       # NW
}

_CHAIN_KEY_MAP: dict[int, int] = {
    pygame.K_LEFT: 0,
    pygame.K_0: 0,
    pygame.K_KP0: 0,
    pygame.K_RIGHT: 1,
    pygame.K_1: 1,
    pygame.K_KP1: 1,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive player
# ══════════════════════════════════════════════════════════════════════════════

class InteractivePlayer:
    def __init__(self, env_id: str, seed: int = 42):
        self.env_id = env_id
        self.seed = seed
        self.env: gymnasium.Env | None = None
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.small_font: pygame.font.Font | None = None
        self.title_font: pygame.font.Font | None = None

        self.total_reward = 0.0
        self.last_reward = 0.0
        self.episode_count = 0
        self.done = False

    # ── setup ────────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.set_caption(f"LPG Play – {self.env_id}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 14)
        self.title_font = pygame.font.SysFont("monospace", 22, bold=True)

    def _create_env(self) -> None:
        kwargs: dict = {}
        if _is_grid_world(self.env_id):
            kwargs["action_mode"] = "move_only"
        self.env = gymnasium.make(self.env_id, **kwargs)

    def _window_size(self) -> tuple[int, int]:
        if _is_grid_world(self.env_id):
            uw = self.env.unwrapped
            w = uw.width * CELL_SIZE
            h = uw.height * CELL_SIZE + HUD_HEIGHT
            return w, h
        return CHAIN_WIDTH, CHAIN_HEIGHT + HUD_HEIGHT

    # ── episode management ───────────────────────────────────────────────

    def _reset_episode(self, new_seed: int | None = None) -> None:
        if new_seed is not None:
            self.env.reset(seed=new_seed)
        else:
            self.env.reset()
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.episode_count += 1
        self.done = False

    # ── rendering: grid world ────────────────────────────────────────────

    def _render_grid_world(self) -> None:
        uw = self.env.unwrapped
        cs = CELL_SIZE
        grid_w, grid_h = uw.width, uw.height

        self.screen.fill(BG_COLOR)

        # ── draw floor and walls from the map ──
        for r in range(grid_h):
            for c in range(grid_w):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                if uw._wall_map[r, c]:
                    pygame.draw.rect(self.screen, DARK_GREY, rect)
                    inner = pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4)
                    pygame.draw.rect(self.screen, BORDER_INNER, inner)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)

        # grid lines on floor cells
        for r in range(grid_h + 1):
            pygame.draw.line(self.screen, LIGHT_GREY, (0, r * cs), (grid_w * cs, r * cs))
        for c in range(grid_w + 1):
            pygame.draw.line(self.screen, LIGHT_GREY, (c * cs, 0), (c * cs, grid_h * cs))

        # ── objects ──
        for i in range(uw.num_objects):
            if uw._object_present[i]:
                r, c = uw._object_positions[i]
                color = GOLD if uw.config.objects[i].reward > 0 else RED
                cx = c * cs + cs // 2
                cy = r * cs + cs // 2
                pygame.draw.circle(self.screen, color, (cx, cy), cs // 3)

        # ── agent ──
        ar, ac = uw._agent_pos
        agent_rect = pygame.Rect(
            ac * cs + cs // 4,
            ar * cs + cs // 4,
            cs // 2,
            cs // 2,
        )
        pygame.draw.rect(self.screen, BLUE, agent_rect)

        # ── HUD ──
        hud_y = grid_h * cs
        self._render_hud(hud_y)

    # ── rendering: chain MDP ─────────────────────────────────────────────

    def _render_chain_mdp(self) -> None:
        """Render chain MDP as a sideways-Y / umbrella layout (matching demo GIF style).

        A single Start cell on the left branches into two horizontal rows
        (one per action a0 / a1) with color-coded progress and terminal rewards.
        """
        uw = self.env.unwrapped
        self.screen.fill(BG_COLOR)

        chain_len = uw._chain_length
        if chain_len is None:
            return

        current_step = uw._current_step
        first_action = uw._first_action
        correct_action = uw._correct_action

        num_cells = chain_len - 1  # cells per row (steps 1..N-1)

        # ── layout constants ──
        margin = 25
        start_cell_w, start_cell_h = 60, 44
        branch_gap = 35
        label_w = 35
        reward_label_w = 50
        cell_gap = 5
        row_gap = 24

        # adaptive cell size
        avail_w = CHAIN_WIDTH - 2 * margin - start_cell_w - branch_gap - label_w - reward_label_w
        max_cell, min_cell = 34, 14
        cell_w = min(max_cell, max(min_cell, (avail_w - (num_cells - 1) * cell_gap) // max(num_cells, 1)))
        cell_h = cell_w

        # vertical positions
        y_title = 12
        y_row0 = y_title + 42           # a0 row
        y_row1 = y_row0 + cell_h + row_gap  # a1 row

        # ── title ──
        title = self.title_font.render(f"Delayed Chain MDP  (length={chain_len})", True, HUD_TEXT)
        self.screen.blit(title, (margin, y_title))

        # ── start cell (centred between the two rows) ──
        start_x = margin
        start_cy = (y_row0 + y_row1 + cell_h) // 2
        start_y = start_cy - start_cell_h // 2
        start_color = BLUE if current_step == 0 and not self.done else MID_GREY
        start_rect = pygame.Rect(start_x, start_y, start_cell_w, start_cell_h)
        pygame.draw.rect(self.screen, start_color, start_rect, border_radius=6)
        lbl_surf = self.small_font.render("Start", True, WHITE)
        self.screen.blit(
            lbl_surf,
            (start_x + (start_cell_w - lbl_surf.get_width()) // 2, start_cy - lbl_surf.get_height() // 2),
        )

        # ── branch lines from start cell to each row ──
        branch_x0 = start_x + start_cell_w
        rows_x0 = branch_x0 + branch_gap
        for y_row in (y_row0, y_row1):
            pygame.draw.line(
                self.screen, (160, 160, 160),
                (branch_x0, start_cy),
                (rows_x0 - 4, y_row + cell_h // 2),
                2,
            )

        # ── row labels (a0 / a1) ──
        for action, y_row in ((0, y_row0), (1, y_row1)):
            lbl = self.small_font.render(f"a{action}", True, MID_GREY)
            self.screen.blit(lbl, (rows_x0, y_row + (cell_h - lbl.get_height()) // 2))

        # ── chain cells (two rows) ──
        cells_x0 = rows_x0 + label_w
        for action, y_row in ((0, y_row0), (1, y_row1)):
            is_chosen = first_action == action
            is_correct = action == correct_action

            for i in range(num_cells):
                x = cells_x0 + i * (cell_w + cell_gap)
                step_idx = i + 1
                is_terminal = step_idx == num_cells

                # pick colour
                if is_chosen and step_idx == current_step and not self.done:
                    color = BLUE                                    # agent here
                elif is_chosen and step_idx < current_step:
                    color = (90, 190, 90) if is_correct else (210, 100, 100)  # trail
                elif not is_chosen:
                    color = (60, 60, 60)                            # unchosen path (faded)
                else:
                    color = (120, 120, 120)                         # future on chosen path

                rect = pygame.Rect(x, y_row, cell_w, cell_h)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

                # gold border on terminal cell
                if is_terminal:
                    pygame.draw.rect(self.screen, GOLD, rect, width=2, border_radius=4)

                # connector line to next cell
                if i < num_cells - 1:
                    nx = x + cell_w
                    pygame.draw.line(
                        self.screen, (100, 100, 100),
                        (nx, y_row + cell_h // 2),
                        (nx + cell_gap, y_row + cell_h // 2),
                    )

        # ── terminal reward labels (+1 / -1) ──
        last_x = cells_x0 + (num_cells - 1) * (cell_w + cell_gap) + cell_w + 8
        for action, y_row in ((0, y_row0), (1, y_row1)):
            is_correct = action == correct_action
            lbl_text = "+1" if is_correct else "-1"
            lbl_color = GREEN if is_correct else RED
            lbl = self.font.render(lbl_text, True, lbl_color)
            self.screen.blit(lbl, (last_x, y_row + (cell_h - lbl.get_height()) // 2))

        # ── info bar ──
        y_info = y_row1 + cell_h + 16
        action_str = f"a{first_action}" if first_action is not None else "?"
        info1 = f"Step: {current_step}/{chain_len}    First action: {action_str}  (correct: a{correct_action})"
        info2 = f"Reward: {self.last_reward:+.1f}    Total: {self.total_reward:+.1f}"
        self.screen.blit(self.font.render(info1, True, HUD_TEXT), (margin, y_info))
        r_color = GREEN if self.last_reward > 0 else RED if self.last_reward < 0 else MID_GREY
        self.screen.blit(self.font.render(info2, True, r_color), (margin, y_info + 22))

        if self.done:
            done_surf = self.font.render("[DONE – press R]", True, GOLD)
            self.screen.blit(done_surf, (margin + 350, y_info + 22))

        self._render_hud(CHAIN_HEIGHT)

    # ── HUD bar ──────────────────────────────────────────────────────────

    def _render_hud(self, y_offset: int) -> None:
        w = self.screen.get_width()
        hud_rect = pygame.Rect(0, y_offset, w, HUD_HEIGHT)
        pygame.draw.rect(self.screen, HUD_BG, hud_rect)
        pygame.draw.line(self.screen, MID_GREY, (0, y_offset), (w, y_offset))

        uw = self.env.unwrapped

        if _is_grid_world(self.env_id):
            step = uw._step_count
            max_steps = uw.config.max_episode_steps
            line1 = (
                f"Step: {step}/{max_steps}  |  "
                f"Reward: {self.last_reward:+.1f}  |  "
                f"Total: {self.total_reward:+.1f}  |  "
                f"Ep: {self.episode_count}"
            )
            line2 = (
                "Move: Arrows/WASD  Diag: Q/E/Z/C  Stay: Space  |  "
                "[R] Reset  [N] New lifetime  [ESC] Menu"
            )
        else:
            step = uw._current_step
            chain_len = uw._chain_length or 0
            line1 = (
                f"Step: {step}/{chain_len}  |  "
                f"Reward: {self.last_reward:+.1f}  |  "
                f"Total: {self.total_reward:+.1f}  |  "
                f"Ep: {self.episode_count}"
            )
            line2 = (
                "[Left/0] Action 0  |  [Right/1] Action 1  |  "
                "[R] Reset  [N] New lifetime  [ESC] Menu"
            )

        if self.done:
            line1 += "  [DONE – press R]"

        surf1 = self.font.render(line1, True, HUD_TEXT)
        surf2 = self.small_font.render(line2, True, MID_GREY)
        self.screen.blit(surf1, (10, y_offset + 8))
        self.screen.blit(surf2, (10, y_offset + 32))

    # ── main loop ────────────────────────────────────────────────────────

    def run(self) -> str:
        """Run the interactive player. Returns ``"menu"`` to go back."""
        self._init_pygame()
        self._create_env()
        w, h = self._window_size()
        self.screen = pygame.display.set_mode((w, h))

        self._reset_episode(new_seed=self.seed)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    pygame.quit()
                    return "quit"

                if event.type != pygame.KEYDOWN:
                    continue

                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if event.key == pygame.K_r:
                    self._reset_episode()
                    continue

                if event.key == pygame.K_n:
                    self._reset_episode(new_seed=random.randint(0, 999_999))
                    continue

                if self.done:
                    continue

                if _is_grid_world(self.env_id):
                    action = _GRID_KEY_MAP.get(event.key)
                else:
                    action = _CHAIN_KEY_MAP.get(event.key)

                if action is not None and self.env.action_space.contains(action):
                    _, reward, terminated, truncated, _ = self.env.step(action)
                    self.last_reward = reward
                    self.total_reward += reward
                    self.done = terminated or truncated

            # draw
            if _is_grid_world(self.env_id):
                self._render_grid_world()
            else:
                self._render_chain_mdp()
            pygame.display.flip()
            self.clock.tick(FPS)

        self.env.close()
        pygame.quit()
        return "menu"


# ══════════════════════════════════════════════════════════════════════════════
#  Environment selection menu
# ══════════════════════════════════════════════════════════════════════════════

def _run_menu() -> str | None:
    """Show a pygame menu and return the chosen environment ID, or ``None``."""
    pygame.init()

    menu_w, menu_h = 700, 620
    screen = pygame.display.set_mode((menu_w, menu_h))
    pygame.display.set_caption("LPG Play – Environment Selection")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("monospace", 26, bold=True)
    group_font = pygame.font.SysFont("monospace", 17, bold=True)
    item_font = pygame.font.SysFont("monospace", 15)
    hint_font = pygame.font.SysFont("monospace", 13)

    # Build flat list with group headers
    flat_items: list[tuple[str, str]] = []  # ("group"|"env", label)
    for group, ids in ENV_GROUPS.items():
        flat_items.append(("group", group))
        for env_id in ids:
            flat_items.append(("env", env_id))

    selectable = [i for i, (t, _) in enumerate(flat_items) if t == "env"]
    sel_pos = 0  # index into selectable

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                if event.key in (pygame.K_UP, pygame.K_k):
                    sel_pos = max(0, sel_pos - 1)
                elif event.key in (pygame.K_DOWN, pygame.K_j):
                    sel_pos = min(len(selectable) - 1, sel_pos + 1)
                elif event.key == pygame.K_RETURN:
                    _, env_id = flat_items[selectable[sel_pos]]
                    pygame.quit()
                    return env_id

        screen.fill(MENU_BG)

        # title
        t = title_font.render("LPG Environment Player", True, WHITE)
        screen.blit(t, (menu_w // 2 - t.get_width() // 2, 20))
        h = hint_font.render(
            "Up/Down to select, Enter to play, ESC to quit", True, MID_GREY
        )
        screen.blit(h, (menu_w // 2 - h.get_width() // 2, 55))

        y = 90
        selected_idx = selectable[sel_pos]
        for i, (item_type, label) in enumerate(flat_items):
            if item_type == "group":
                y += 8
                surf = group_font.render(f"-- {label} --", True, GOLD)
                screen.blit(surf, (50, y))
                y += 26
            else:
                short = label.split("/", 1)[-1]
                if i == selected_idx:
                    hl = pygame.Rect(40, y - 2, menu_w - 80, 22)
                    pygame.draw.rect(screen, MENU_HIGHLIGHT, hl, border_radius=4)
                    surf = item_font.render(f"> {short}", True, WHITE)
                else:
                    surf = item_font.render(f"  {short}", True, HUD_TEXT)
                screen.blit(surf, (50, y))
                y += 24

        pygame.display.flip()
        clock.tick(30)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive LPG environment player")
    parser.add_argument(
        "env_id", nargs="?", default=None, help="Environment ID (omit for menu)"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all available environment IDs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Initial random seed (default: 42)"
    )
    args = parser.parse_args()

    if args.list:
        print("Available environments:")
        for group, ids in ENV_GROUPS.items():
            print(f"\n  {group}:")
            for eid in ids:
                print(f"    {eid}")
        return

    env_id: str | None = args.env_id

    while True:
        if env_id is None:
            env_id = _run_menu()
            if env_id is None:
                break

        if env_id not in ALL_ENV_IDS:
            print(f"Unknown environment: {env_id}")
            print("Use --list to see available environments.")
            return

        result = InteractivePlayer(env_id, seed=args.seed).run()
        if result == "quit":
            break
        env_id = None  # back to menu


if __name__ == "__main__":
    main()
