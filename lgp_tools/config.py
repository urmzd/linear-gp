"""Centralized configuration for lgp_tools."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiscoveredConfig:
    """A discovered experiment configuration."""

    name: str
    config_path: Path


def get_configs_dir() -> Path:
    """Get config directory from environment or default."""
    return Path(os.environ.get("LGP_CONFIGS_DIR", "configs"))


def discover_configs() -> list[DiscoveredConfig]:
    """Discover all experiment configs with default.toml."""
    configs_dir = get_configs_dir()
    if not configs_dir.exists():
        return []

    configs = []
    for item in sorted(configs_dir.iterdir()):
        if item.is_dir():
            config_path = item / "default.toml"
            if config_path.exists():
                configs.append(DiscoveredConfig(name=item.name, config_path=config_path))
    return configs


def find_config(name: str) -> DiscoveredConfig:
    """Find a specific config by name."""
    configs_dir = get_configs_dir()
    config_path = configs_dir / name / "default.toml"
    if not config_path.exists():
        raise ValueError(f"Config '{name}' not found at {config_path}")
    return DiscoveredConfig(name=name, config_path=config_path)


# Database storage
STORAGE = "postgresql://user:password@localhost:5432/database"

# Pruning thresholds for hyperparameter search
PRUNE_THRESHOLDS = {
    "cart": 400,
    "iris": 0.9,
    "mountain": -150,
    "default": 0,
}
