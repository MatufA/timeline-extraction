"""Configuration management for timeline extraction experiments."""

import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from timeline_extraction.utils import load_json, save_json, ensure_directory


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    name: str
    provider: str
    api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.0
    device: str = "cuda:0"
    cache_dir: str = "./cache"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""

    method: str = "zero-shot"  # zero-shot, few-shot
    mode: str = "multi"  # pair, multi, comb
    use_vague: bool = True
    parser_type: str = "json"  # json, label
    overwrite: bool = False
    skip_model_eval: bool = False


@dataclass
class DataConfig:
    """Configuration for data settings."""

    data_path: str = "./data"
    results_path: str = "./results"
    cache_path: str = "./cache"
    datasets: Dict[str, Dict[str, str]] = None


@dataclass
class ExperimentConfig:
    """Configuration for experiment settings."""

    name: str
    description: str = ""
    output_dir: str = "./experiments"
    save_metadata: bool = True
    save_prompts: bool = True
    save_responses: bool = True
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ConfigManager:
    """Manager for experiment configurations."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.model_configs: Dict[str, ModelConfig] = {}
        self.evaluation_config = EvaluationConfig()
        self.data_config = DataConfig()
        self.experiment_config: Optional[ExperimentConfig] = None

        if self.config_path and self.config_path.exists():
            self.load_config()

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            if self.config_path.suffix in [".yaml", ".yml"]:
                with open(self.config_path, "r") as f:
                    config_data = yaml.safe_load(f)
            else:
                config_data = load_json(self.config_path)

            self._parse_config(config_data)
            logging.info(f"Configuration loaded from {self.config_path}")

        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path:
            raise ValueError("No configuration path specified")

        config_data = self._to_dict()

        try:
            if self.config_path.suffix in [".yaml", ".yml"]:
                with open(self.config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                save_json(config_data, self.config_path)

            logging.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            raise

    def _parse_config(self, config_data: Dict[str, Any]) -> None:
        """Parse configuration data into config objects."""
        # Parse model configurations
        if "models" in config_data:
            for provider, model_data in config_data["models"].items():
                model_config = ModelConfig(
                    name=model_data.get("default_model", ""),
                    provider=provider,
                    api_key=model_data.get("api_key"),
                    max_tokens=model_data.get("max_tokens", 2000),
                    temperature=model_data.get("temperature", 0.0),
                    device=model_data.get("device", "cuda:0"),
                    cache_dir=model_data.get("cache_dir", "./cache"),
                )
                self.model_configs[provider] = model_config

        # Parse evaluation configuration
        if "evaluation" in config_data:
            eval_data = config_data["evaluation"]
            self.evaluation_config = EvaluationConfig(
                method=eval_data.get("default_method", "zero-shot"),
                mode=eval_data.get("default_mode", "multi"),
                use_vague=eval_data.get("use_vague", True),
                parser_type=eval_data.get("parser_type", "json"),
                overwrite=eval_data.get("overwrite", False),
                skip_model_eval=eval_data.get("skip_model_eval", False),
            )

        # Parse data configuration
        if "data" in config_data:
            data_data = config_data["data"]
            self.data_config = DataConfig(
                data_path=data_data.get("data_path", "./data"),
                results_path=data_data.get("results_path", "./results"),
                cache_path=data_data.get("cache_path", "./cache"),
                datasets=data_data.get("datasets", {}),
            )

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "models": {
                provider: asdict(config)
                for provider, config in self.model_configs.items()
            },
            "evaluation": asdict(self.evaluation_config),
            "data": asdict(self.data_config),
        }

        if self.experiment_config:
            config_dict["experiment"] = asdict(self.experiment_config)

        return config_dict

    def create_experiment(
        self, name: str, description: str = "", output_dir: Optional[str] = None
    ) -> ExperimentConfig:
        """Create a new experiment configuration.

        Args:
            name: Experiment name
            description: Experiment description
            output_dir: Output directory for experiment

        Returns:
            Experiment configuration
        """
        if output_dir is None:
            output_dir = (
                f"./experiments/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        self.experiment_config = ExperimentConfig(
            name=name, description=description, output_dir=output_dir
        )

        # Ensure output directory exists
        ensure_directory(self.experiment_config.output_dir)

        logging.info(f"Created experiment: {name}")
        return self.experiment_config

    def get_model_config(self, provider: str) -> Optional[ModelConfig]:
        """Get model configuration for provider.

        Args:
            provider: Model provider name

        Returns:
            Model configuration or None
        """
        return self.model_configs.get(provider)

    def add_model_config(self, provider: str, model_config: ModelConfig) -> None:
        """Add model configuration.

        Args:
            provider: Model provider name
            model_config: Model configuration
        """
        self.model_configs[provider] = model_config
        logging.info(f"Added model configuration for {provider}")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation issues
        """
        issues = []

        # Validate model configurations
        for provider, config in self.model_configs.items():
            if not config.name:
                issues.append(f"Model name not specified for {provider}")
            if not config.api_key and provider in [
                "openai",
                "google",
                "together",
                "groq",
            ]:
                issues.append(f"API key not specified for {provider}")

        # Validate evaluation configuration
        if self.evaluation_config.method not in ["zero-shot", "few-shot"]:
            issues.append(f"Invalid evaluation method: {self.evaluation_config.method}")

        if self.evaluation_config.mode not in ["pair", "multi", "comb"]:
            issues.append(f"Invalid evaluation mode: {self.evaluation_config.mode}")

        if self.evaluation_config.parser_type not in ["json", "label"]:
            issues.append(f"Invalid parser type: {self.evaluation_config.parser_type}")

        # Validate data configuration
        if not Path(self.data_config.data_path).exists():
            issues.append(f"Data path does not exist: {self.data_config.data_path}")

        return issues

    def get_experiment_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata.

        Returns:
            Dictionary of experiment metadata
        """
        if not self.experiment_config:
            return {}

        return {
            "name": self.experiment_config.name,
            "description": self.experiment_config.description,
            "timestamp": self.experiment_config.timestamp,
            "output_dir": self.experiment_config.output_dir,
            "model_configs": {
                provider: asdict(config)
                for provider, config in self.model_configs.items()
            },
            "evaluation_config": asdict(self.evaluation_config),
            "data_config": asdict(self.data_config),
        }


def create_default_config() -> Dict[str, Any]:
    """Create default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "models": {
            "openai": {
                "api_key": "your_openai_key_here",
                "default_model": "gpt-4",
                "max_tokens": 2000,
                "temperature": 0.0,
            },
            "huggingface": {"cache_dir": "./cache", "device": "cuda:0"},
            "google": {
                "api_key": "your_google_key_here",
                "default_model": "gemini-pro",
                "max_output_tokens": 2000,
            },
            "together": {
                "api_key": "your_together_key_here",
                "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            },
            "groq": {
                "api_key": "your_groq_key_here",
                "default_model": "llama3-8b-8192",
            },
        },
        "evaluation": {
            "default_method": "zero-shot",
            "default_mode": "multi",
            "use_vague": True,
            "parser_type": "json",
            "overwrite": False,
            "skip_model_eval": False,
        },
        "data": {
            "data_path": "./data",
            "results_path": "./results",
            "cache_path": "./cache",
            "datasets": {
                "matres": {"path": "./data/MATRES", "gold_file": "platinum.txt"},
                "timebank": {
                    "path": "./data/MATRES/raw/TBAQ-cleaned/TimeBank",
                    "gold_file": "timebank.txt",
                },
                "aquaint": {
                    "path": "./data/MATRES/raw/TBAQ-cleaned/AQUAINT",
                    "gold_file": "aquaint.txt",
                },
            },
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,
        },
        "experiments": {
            "output_dir": "./experiments",
            "save_metadata": True,
            "save_prompts": True,
            "save_responses": True,
        },
    }
