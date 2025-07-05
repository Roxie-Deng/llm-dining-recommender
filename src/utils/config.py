import yaml
import os

def load_config(config_path="configs/data_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_path(config, *keys):
    """Recursively get a path from config using keys."""
    d = config
    for k in keys:
        d = d[k]
    return d 