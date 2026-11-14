"""Load project settings from config.yaml."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return config.yaml parsed into a dict."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
