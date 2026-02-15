"""Delayed Chain MDP configurations from the LPG paper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainMDPConfig:
    """Configuration for a Delayed Chain MDP variant.

    Attributes:
        name: Human-readable variant name.
        chain_length_min: Minimum chain length (sampled per lifetime).
        chain_length_max: Maximum chain length (sampled per lifetime).
        noisy_rewards: Whether intermediate states have random +1/-1 rewards.
        distractor_bits: Number of noisy observation bits (0 for standard).
    """

    name: str
    chain_length_min: int
    chain_length_max: int
    noisy_rewards: bool
    distractor_bits: int = 0


# ---------------------------------------------------------------------------
# Delayed Chain MDP variants (Section A.3)
# ---------------------------------------------------------------------------

CHAIN_CONFIGS: dict[str, ChainMDPConfig] = {
    "short": ChainMDPConfig(
        name="Short",
        chain_length_min=5,
        chain_length_max=30,
        noisy_rewards=False,
    ),
    "short_noisy": ChainMDPConfig(
        name="ShortNoisy",
        chain_length_min=5,
        chain_length_max=30,
        noisy_rewards=True,
    ),
    "long": ChainMDPConfig(
        name="Long",
        chain_length_min=5,
        chain_length_max=50,
        noisy_rewards=False,
    ),
    "long_noisy": ChainMDPConfig(
        name="LongNoisy",
        chain_length_min=5,
        chain_length_max=50,
        noisy_rewards=True,
    ),
    "distractor": ChainMDPConfig(
        name="Distractor",
        chain_length_min=5,
        chain_length_max=30,
        noisy_rewards=False,
        distractor_bits=20,
    ),
}
