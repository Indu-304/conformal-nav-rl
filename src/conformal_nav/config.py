from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name_or_path: str) -> dict:
    """Load a YAML config by scenario name (e.g. 'intersection') or explicit path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
