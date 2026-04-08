"""
Centralized YAML config loader.

Loads configuration files from the `configs/` directory located at the
project root (the directory containing this file). Each loader function
reads exactly one YAML file and returns the parsed mapping.

Rules enforced:
- Uses yaml.safe_load
- Resolves paths relative to project root (this file's directory)
- Raises FileNotFoundError if a config file is missing
- Does not supply defaults or mutate loaded values
"""
from typing import Dict, Any
import os
import yaml


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "configs")


def _load_yaml_file(filename: str) -> Dict[str, Any]:
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        parsed = yaml.safe_load(fh)
    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration file {path} does not contain a YAML mapping (dict).")
    return parsed


def load_model_config() -> Dict[str, Any]:
    """Load configs/model_config.yaml and return parsed dict."""
    return _load_yaml_file("model_config.yaml")


def load_thresholds() -> Dict[str, Any]:
    """Load configs/thresholds.yaml and return parsed dict."""
    return _load_yaml_file("thresholds.yaml")


def load_intervention_limits() -> Dict[str, Any]:
    """Load configs/intervention_limits.yaml and return parsed dict."""
    return _load_yaml_file("intervention_limits.yaml")


if __name__ == "__main__":
    try:
        mc = load_model_config()
        th = load_thresholds()
        il = load_intervention_limits()
        print("Loaded configs successfully. Keys:")
        print("model_config:", list(mc.keys()))
        print("thresholds:", list(th.keys()))
        print("intervention_limits:", list(il.keys()))
    except Exception as e:
        print("Error loading configs:", e)
        raise

