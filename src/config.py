"""
Configuration module for YAML-based parameter management.

Handles reading/writing of config files with parameter ranges
and multi-file section support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import random


# Parameter range definitions (破綻しない範囲)
PARAM_RANGES: Dict[str, Dict[str, float]] = {
    'L1': {'min': 0.1, 'max': 2.0, 'description': 'top_inner_y'},
    'L2': {'min': 0.1, 'max': 2.0, 'description': 'top_outer_y'},
    'L3': {'min': 0.1, 'max': 2.0, 'description': 'top_z'},
    'L4': {'min': 0.1, 'max': 1.0, 'description': 'upper_step_y'},
    'L5': {'min': 0.1, 'max': 2.0, 'description': 'outer_y'},
    'L6': {'min': 0.1, 'max': 2.0, 'description': 'outer_z'},
    'L7': {'min': 0.1, 'max': 2.0, 'description': 'lower_step_z'},
    'L8': {'min': 0.1, 'max': 2.0, 'description': 'inner_shelf_y'},
    'L9': {'min': 0.1, 'max': 1.5, 'description': 'inner_shelf_z'},
    'L10': {'min': 0.3, 'max': 3.0, 'description': 'bottom_z'},
    'R1': {'min': 0.05, 'max': 0.5, 'description': 'fillet_inner'},
    'R2': {'min': 0.05, 'max': 0.5, 'description': 'fillet_bottom'},
}


@dataclass
class ParameterConfig:
    """Single parameter configuration."""
    value: float
    min: float
    max: float
    description: str
    
    def to_dict(self) -> dict:
        return {
            'description': self.description,
            'value': self.value,
            'min': self.min,
            'max': self.max,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ParameterConfig':
        return cls(
            value=d['value'],
            min=d['min'],
            max=d['max'],
            description=d.get('description', ''),
        )
    
    def random_value(self) -> float:
        """Generate random value within range."""
        return random.uniform(self.min, self.max)
    
    def clamp(self, value: float) -> float:
        """Clamp value to range."""
        return max(self.min, min(self.max, value))


@dataclass
class FileConfig:
    """Configuration for a single STEP file."""
    filename: str
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)
    min_thickness: float = 0.2
    
    def to_dict(self) -> dict:
        return {
            'parameters': {k: v.to_dict() for k, v in self.parameters.items()},
            'constraints': {'min_thickness': self.min_thickness},
        }
    
    @classmethod
    def from_dict(cls, filename: str, d: dict) -> 'FileConfig':
        params = {}
        for k, v in d.get('parameters', {}).items():
            params[k] = ParameterConfig.from_dict(v)
        
        constraints = d.get('constraints', {})
        min_thickness = constraints.get('min_thickness', 0.2)
        
        return cls(filename=filename, parameters=params, min_thickness=min_thickness)
    
    def get_param_dict(self) -> dict:
        """Get parameter values as simple dict."""
        return {k: v.value for k, v in self.parameters.items()}
    
    def randomize_parameters(self) -> None:
        """Randomize all parameters within their ranges."""
        for param in self.parameters.values():
            param.value = param.random_value()


@dataclass
class Config:
    """Main configuration container for multiple files."""
    version: str = "1.0"
    files: Dict[str, FileConfig] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'files': {k: v.to_dict() for k, v in self.files.items()},
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        files = {}
        for filename, file_data in d.get('files', {}).items():
            files[filename] = FileConfig.from_dict(filename, file_data)
        return cls(version=d.get('version', '1.0'), files=files)
    
    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, 
                     default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load config from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def add_file(self, filename: str, params: dict) -> FileConfig:
        """Add or update file configuration."""
        file_config = create_file_config(filename, params)
        self.files[filename] = file_config
        return file_config


def create_file_config(filename: str, current_params: dict) -> FileConfig:
    """Create FileConfig with current values and ranges from PARAM_RANGES."""
    parameters = {}
    
    for name, ranges in PARAM_RANGES.items():
        value = current_params.get(name, (ranges['min'] + ranges['max']) / 2)
        parameters[name] = ParameterConfig(
            value=value,
            min=ranges['min'],
            max=ranges['max'],
            description=ranges['description'],
        )
    
    return FileConfig(filename=filename, parameters=parameters)


def create_default_config(step_files: List[Path]) -> Config:
    """Create default config for list of STEP files."""
    from .geometry import REFERENCE_PARAMS
    
    config = Config()
    default_params = REFERENCE_PARAMS.to_dict()
    
    for step_file in step_files:
        config.add_file(step_file.name, default_params)
    
    return config


def get_config_path() -> Path:
    """Get default config file path."""
    return Path('config.yaml')
