"""Configuration management for timeline extraction experiments."""

import yaml
import os
import math
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime

from timeline_extraction.utils import load_json, save_json, ensure_directory


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    name: str
    provider: str
    api_key: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.0
    device: str = "cpu"
    cache_dir: str = "./cache"
    n_trails: int = 5
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None
    cli_timeout: float = 600

    def __post_init__(self):
        if self.max_tokens < 1 or self.n_trails < 1:
            raise ValueError("max_tokens and n_trails must be positive")
        if self.provider not in {"openai", "google", "groq", "together", "huggingface", "antigravity", "codex", "claude"}:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if self.provider in {"antigravity", "codex", "claude"}:
            if self.api_key or self.base_url:
                raise ValueError(f"{self.provider} uses CLI account login, not api_key or base_url")
            if self.temperature != 0.0:
                raise ValueError(f"{self.provider} CLI does not support temperature")
            if not math.isfinite(self.cli_timeout) or self.cli_timeout <= 0:
                raise ValueError("cli_timeout must be a finite positive number of seconds")
        from timeline_extraction.models._reasoning import reasoning_options
        reasoning_options(self.provider, self.name, self.reasoning_effort)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""

    method: str = "zero-shot"  # zero-shot, few-shot
    mode: str = "pair"  # pair, multi, comb
    use_vague: bool = True
    parser_type: Optional[str] = None  # infer from mode unless explicitly set
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


@dataclass
class ResearchConfig:
    """Runnable experiment selection by research question."""

    goal: str = "classification"
    providers: List[str] = field(default_factory=lambda: ["openai"])
    methods: List[str] = field(default_factory=lambda: ["zero-shot"])
    dataset: str = "te3-platinum"

    def __post_init__(self):
        if self.goal not in {"classification", "consistency", "comparison"}:
            raise ValueError(f"Invalid research goal: {self.goal}")
        if not self.providers:
            raise ValueError("Select at least one provider")
        if not self.methods or any(m not in {"zero-shot", "few-shot"} for m in self.methods):
            raise ValueError("Research methods must be zero-shot or few-shot")

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
        self.research_config = ResearchConfig()

        if self.config_path:
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
            self.model_configs = {}
            for provider, model_data in config_data["models"].items():
                model_config = ModelConfig(
                    name=model_data.get("default_model", model_data.get("name", "")),
                    provider=provider,
                    api_key=model_data.get("api_key"),
                    max_tokens=model_data.get("max_tokens", model_data.get("max_output_tokens", 8192)),
                    temperature=model_data.get("temperature", 0.0),
                    device=model_data.get("device", "cpu"),
                    cache_dir=model_data.get("cache_dir", "./cache"),
                    n_trails=model_data.get("n_trails", 5),
                    base_url=model_data.get("base_url"),
                    reasoning_effort=model_data.get("reasoning_effort"),
                    cli_timeout=model_data.get("cli_timeout", 600),
                )
                self.model_configs[provider] = model_config

        # Parse evaluation configuration
        if "evaluation" in config_data:
            eval_data = config_data["evaluation"]
            self.evaluation_config = EvaluationConfig(
                method=eval_data.get("default_method", eval_data.get("method", "zero-shot")),
                mode=eval_data.get("default_mode", eval_data.get("mode", "pair")),
                use_vague=eval_data.get("use_vague", True),
                parser_type=eval_data.get("parser_type"),
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
        if "research" in config_data:
            self.research_config = ResearchConfig(**config_data["research"])
        if "experiment" in config_data:
            self.experiment_config = ExperimentConfig(**config_data["experiment"])

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        model_configs = {}
        for provider, config in self.model_configs.items():
            settings = asdict(config)
            settings["default_model"] = settings.pop("name")
            settings.pop("provider")
            model_configs[provider] = settings
        evaluation = asdict(self.evaluation_config)
        evaluation["default_method"] = evaluation.pop("method")
        evaluation["default_mode"] = evaluation.pop("mode")
        config_dict = {
            "models": model_configs,
            "evaluation": evaluation,
            "data": asdict(self.data_config),
            "research": asdict(self.research_config),
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

    def resolve_model_config(
        self, model_name: Optional[str] = None, provider: Optional[str] = None
    ) -> ModelConfig:
        """Resolve explicit provider, exact configured model, then unambiguous prefixes."""
        if provider is None:
            matches = [c for c in self.model_configs.values() if c.name == model_name]
            if len(matches) > 1:
                raise ValueError(f"Model {model_name} is configured under multiple providers; select --provider")
            if matches:
                return replace(matches[0])
            name = (model_name or "").lower()
            if name.startswith(("gpt-", "o1", "o3", "o4")):
                provider = "openai"
            elif name.startswith("gemini-"):
                provider = "google"
            else:
                provider = "huggingface"
        defaults = create_default_config()["models"].get(provider)
        if defaults is None:
            raise ValueError(f"Unsupported provider: {provider}")
        configured = self.model_configs.get(provider)
        if configured:
            return replace(configured, name=model_name or configured.name or defaults["default_model"])
        name = model_name or defaults.get("default_model")
        if not name:
            raise ValueError(f"Select a model for {provider}")
        settings = {k: v for k, v in defaults.items() if k != "default_model"}
        return ModelConfig(name=name, provider=provider, **settings)

    def merge_config(self, config_data: Dict[str, Any]) -> None:
        """Merge supplied sections into the effective configuration."""
        def merge(target, supplied):
            for key, value in supplied.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value

        merged = self._to_dict()
        merge(merged, config_data)
        self._parse_config(merged)

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
            env_keys = {
                "openai": ("OPENAI_API_KEY",),
                "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                "together": ("TOGETHER_API_KEY",),
                "groq": ("GROQ_API_KEY",),
            }
            if provider in env_keys and not config.api_key and not any(
                os.getenv(key) for key in env_keys[provider]
            ):
                issues.append(f"API key not specified for {provider}")

        # Validate evaluation configuration
        if self.evaluation_config.method not in ["zero-shot", "few-shot"]:
            issues.append(f"Invalid evaluation method: {self.evaluation_config.method}")

        if self.evaluation_config.mode not in ["pair", "multi", "comb"]:
            issues.append(f"Invalid evaluation mode: {self.evaluation_config.mode}")

        if self.evaluation_config.parser_type not in [None, "json", "label"]:
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
                provider: {k: v for k, v in asdict(config).items() if k != "api_key"}
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
                "api_key": None,
                "default_model": "gpt-6-luna",
                "max_tokens": 8192,
                "temperature": 0.0,
            },
            "huggingface": {
                "default_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "cache_dir": "./cache",
                "device": "cpu",
            },
            "google": {
                "api_key": None,
                "default_model": "gemini-3.8-flash",
                "max_tokens": 8192,
            },
            "together": {
                "api_key": None,
                "default_model": "openai/gpt-oss-120b",
                "max_tokens": 8192,
            },
            "groq": {
                "api_key": None,
                "default_model": "openai/gpt-oss-20b",
                "max_tokens": 8192,
            },
            "antigravity": {
                "default_model": "gemini-4-argon-high",
                "reasoning_effort": "high",
                "n_trails": 1,
            },
            "codex": {
                "default_model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "n_trails": 1,
            },
            "claude": {
                "default_model": "sonnet",
                "reasoning_effort": "high",
                "n_trails": 1,
            },
        },
        "research": {
            "goal": "classification",
            "providers": ["openai"],
            "methods": ["zero-shot"],
            "dataset": "te3-platinum",
        },
        "evaluation": {
            "default_method": "zero-shot",
            "default_mode": "pair",
            "use_vague": True,
            "overwrite": False,
            "skip_model_eval": False,
        },
        "data": {
            "data_path": "./data",
            "results_path": "./results",
            "cache_path": "./cache",
            "datasets": {},
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
