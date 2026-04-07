"""I/O utilities: configuration loading, result serialisation, and checkpoint management."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import OmegaConf


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return a plain dictionary.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as a pretty-printed JSON file.

    Parameters
    ----------
    data : dict
        Data to serialise.
    path : str
        Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config_artifact(config: Any, output_dir: str) -> None:
    """Save a structured config (dataclass or OmegaConf) as both YAML and JSON.

    Parameters
    ----------
    config : Any
        An OmegaConf DictConfig or a dataclass instance.
    output_dir : str
        Directory to write ``config.yaml`` and ``config.json`` into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(config, "__dataclass_fields__"):
        cfg = OmegaConf.structured(config)
    else:
        cfg = config

    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    plain = OmegaConf.to_container(cfg, resolve=True)
    save_json(plain, str(out / "config.json"))
