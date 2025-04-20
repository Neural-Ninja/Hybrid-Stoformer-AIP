import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

class Config:
    """
    Configuration class for Stoformer model, training, and testing
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with default values or from a YAML file
        
        Args:
            config_path: Path to the YAML configuration file (optional)
        """
        # Set default values
        self.config = self._get_default_config()
        
        # Load from YAML file if provided
        if config_path:
            self._load_from_yaml(config_path)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values
        
        Returns:
            Dictionary with default configuration
        """
        return {
            # Model parameters
            'model': {
                'name': 'stoformer',
                'img_size': 256,
                'embed_dim': 32,
                'depths': [1, 2, 8, 8],
                'num_heads': [1, 2, 4, 8],
                'window_size': 8,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'patch_norm': True
            },
            
            # Data parameters
            'data': {
                'task': 'denoising',  # 'denoising', 'deraining', 'deblurring'
                'clean_dir': None,
                'degraded_dir': None,
                'val_clean_dir': None,
                'val_degraded_dir': None,
                'test_clean_dir': None,
                'test_degraded_dir': None,
                'patch_size': 256,
                'batch_size': 4,
                'val_batch_size': 1,
                'test_batch_size': 1,
                'num_workers': 4,
                'sigma': [15, 25, 50]  # Noise levels for denoising
            },
            
            # Training parameters
            'train': {
                'seed': 42,
                'lr': 3e-4,
                'min_lr': 1e-6,
                'epochs': 300,
                'save_frequency': 10,
                'resume': False,
                'resume_path': None,
                'save_dir': './checkpoints',
                'find_lr': False
            },
            
            # Testing parameters
            'test': {
                'checkpoint_path': None,
                'save_dir': './results',
                'save_images': True
            },
            
            # Loss parameters
            'loss': {
                'name': 'charbonnier',
                'eps': 1e-3
            },
            
            # Optimizer parameters
            'optimizer': {
                'name': 'adam',
                'betas': [0.9, 0.999],
                'weight_decay': 0.0
            },
            
            # Scheduler parameters
            'scheduler': {
                'name': 'cosine',
                'T_max': None,  # Will be set to num_epochs by default
                'eta_min': 1e-6
            }
        }
    
    def _load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from a YAML file
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update config with values from YAML
        self._update_config(self.config, yaml_config)
    
    def _update_config(self, config: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Recursively update configuration dictionary
        
        Args:
            config: Configuration dictionary to update
            update: Dictionary with new values
        """
        for key, value in update.items():
            if key in config and isinstance(value, dict) and isinstance(config[key], dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.window_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.window_size')
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration from command line arguments
        
        Args:
            args: Command line arguments
        """
        # Convert args namespace to dictionary
        args_dict = vars(args)
        
        # Map args to config keys
        arg_map = {
            # Data parameters
            'task': 'data.task',
            'clean_dir': 'data.clean_dir',
            'degraded_dir': 'data.degraded_dir',
            'val_clean_dir': 'data.val_clean_dir',
            'val_degraded_dir': 'data.val_degraded_dir',
            'patch_size': 'data.patch_size',
            'batch_size': 'data.batch_size',
            'sigma': 'data.sigma',
            'num_workers': 'data.num_workers',
            
            # Model parameters
            'window_size': 'model.window_size',
            'img_size': 'model.img_size',
            
            # Training parameters
            'lr': 'train.lr',
            'epochs': 'train.epochs',
            'save_dir': 'train.save_dir',
            'save_frequency': 'train.save_frequency',
            'seed': 'train.seed',
            'resume': 'train.resume',
            'resume_path': 'train.resume_path',
            'find_lr': 'train.find_lr',
            
            # Testing parameters
            'checkpoint_path': 'test.checkpoint_path',
            'save_dir': 'test.save_dir'
        }
        
        # Update config with args values
        for arg_name, config_key in arg_map.items():
            if arg_name in args_dict and args_dict[arg_name] is not None:
                self.set(config_key, args_dict[arg_name])
    
    def save(self, save_path: str) -> None:
        """
        Save configuration to a YAML file
        
        Args:
            save_path: Path to save the configuration
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration to YAML file
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return yaml.dump(self.config, default_flow_style=False)


def get_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    """
    Get configuration with specified parameters
    
    Args:
        config_path: Path to YAML configuration file (optional)
        args: Command line arguments (optional)
        
    Returns:
        Configuration object
    """
    config = Config(config_path)
    
    if args:
        config.update_from_args(args)
    
    return config 