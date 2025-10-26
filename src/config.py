"""Global configuration helpers for AIS-MDA.
- Load YAML/JSON experiment configs
- Provide dataclasses for common hyperparameters
"""
from dataclasses import dataclass
from pathlib import Path
import yaml
import json

def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        return json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

@dataclass
class TrainConfig:
    task: str = "trajectory"
    window: int = 64
    horizon: int = 12
    features: list[str] = None
    model: dict = None
    loss: str = "huber"
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 20
