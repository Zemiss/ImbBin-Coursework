"""Config management for training and testing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    return config


def _resolve_paths(config_dict: Dict[str, Any], main_dir: Path) -> Dict[str, Any]:
    """Convert paths relative to main directory to absolute paths.
    
    Args:
        config_dict: Configuration dictionary (e.g., config['train'])
        main_dir: Path to main directory
        
    Returns:
        Updated dictionary with resolved paths
    """
    path_keys = ['train_data', 'test_data', 'model_path', 'prediction_path']
    
    for key in path_keys:
        if key in config_dict and config_dict[key] is not None:
            path = Path(config_dict[key])
            if not path.is_absolute():
                # Convert relative path (relative to main) to absolute path
                config_dict[key] = str((main_dir / path).resolve())
    
    return config_dict


def merge_config_with_args(
    config: Dict[str, Any], 
    args: argparse.Namespace, 
    config_key: str,
    main_dir: Path | None = None
) -> argparse.Namespace:
    """Merge YAML config with command-line arguments.
    
    Command-line arguments take precedence over config file values.
    Only arguments that were explicitly provided on command line override config.
    
    Args:
        config: Loaded YAML configuration dictionary
        args: Parsed command-line arguments
        config_key: Key in config dict for this script (e.g., 'train' or 'test')
        main_dir: Path to main directory for resolving relative paths
        
    Returns:
        Updated argparse.Namespace with merged values
    """
    if config_key not in config:
        return args
    
    config_values = config[config_key]
    
    # Resolve paths relative to main directory if provided
    if main_dir:
        config_values = _resolve_paths(config_values, main_dir)
    
    # For each key in the config section
    for key, value in config_values.items():
        # Only set if the argument doesn't already have this attribute
        # or if it has the default value (None)
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
        else:
            # If CLI arg is a path, resolve it too
            if key in ['train_data', 'test_data', 'model_path', 'prediction_path']:
                cli_value = getattr(args, key)
                if cli_value and not Path(cli_value).is_absolute():
                    # Resolve CLI path relative to main directory
                    if main_dir:
                        setattr(args, key, str((main_dir / cli_value).resolve()))
    
    return args


def get_default_config_path() -> Path:
    """Get path to default config file.
    
    Returns:
        Path to default.yaml in configs directory
    """
    root = Path(__file__).resolve().parent.parent  # main/src -> main
    config_path = root / "configs" / "default.yaml"
    return config_path


def get_main_dir() -> Path:
    """Get path to main directory.
    
    Returns:
        Absolute path to main directory
    """
    return Path(__file__).resolve().parent.parent  # main/src -> main


def load_config(
    config_key: str,
    cli_args: argparse.Namespace | None = None,
    config_path: str | Path | None = None
) -> argparse.Namespace:
    """Load configuration from file and merge with CLI arguments.
    
    Paths in the config file are relative to the main directory and will be
    converted to absolute paths.
    
    Args:
        config_key: Key in config dict ('train' or 'test')
        cli_args: Parsed CLI arguments (will override config values)
        config_path: Path to config file (default: configs/default.yaml)
        
    Returns:
        Merged configuration as argparse.Namespace with resolved absolute paths
    """
    if config_path is None:
        config_path = get_default_config_path()
    
    main_dir = get_main_dir()
    
    # Convert config_path to absolute if relative
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (main_dir / config_path).resolve()
    
    config = load_yaml_config(config_path)
    
    if cli_args is None:
        cli_args = argparse.Namespace()
    
    return merge_config_with_args(config, cli_args, config_key, main_dir)
