"""Timeline Extraction Pipeline - Elegant and Clean Implementation

This module provides a clean, configurable pipeline for temporal relation extraction
using Large Language Models with support for multiple models and evaluation methods.
"""

import os
import gc
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, replace, asdict
from datetime import datetime
import json

import pandas as pd
import torch
from tqdm.auto import tqdm

from timeline_extraction.data.postprocessing import majority_vote_decision_new
from timeline_extraction.data.preprocessing import load_data
from timeline_extraction.graph import Graph
from timeline_extraction.metrics import summary_results
from timeline_extraction.models.LLModel import LLModel, LabelParser, JsonParser
from timeline_extraction.prompts.Prompt import Prompt, PairwisePrompt, MultiEvents
from timeline_extraction.config import ConfigManager, ModelConfig, EvaluationConfig, ExperimentConfig as ConfigExperimentConfig
from timeline_extraction.exceptions import ModelError, DataError
from timeline_extraction.tracking import ExperimentTracker, TrackingConfig

# Configure logging
logger = logging.getLogger(__name__)



@dataclass
class PipelineExperimentConfig:
    """Pipeline-specific experiment configuration that extends the base config classes."""
    
    # Core experiment parameters
    model_name: str
    method: str  # zero-shot, few-shot
    mode: str  # pair, multi, comb
    data_name: str  # te3-platinum, timebank, aquaint
    
    # Evaluation settings (reuse from EvaluationConfig)
    use_vague: bool = True
    parser_type: Optional[str] = None  # Inferred from mode unless explicitly selected
    overwrite: bool = False
    skip_model_eval: bool = False
    
    # Pipeline-specific settings
    full_context: bool = True
    suffix_path: str = "completion"
    prompt_params: List[str] = field(default_factory=lambda: ["text", "relations"])
    gpu_device: int = 0
    output_dir: Optional[str] = None
    research_goal: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.method not in ["zero-shot", "few-shot"]:
            raise ValueError(f"Invalid method: {self.method}")
        if self.mode not in ["pair", "multi", "comb"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.parser_type is None:
            self.parser_type = "json" if self.mode == "multi" else "label"
        if self.parser_type not in ["json", "label"]:
            raise ValueError(f"Invalid parser_type: {self.parser_type}")
    
    def to_evaluation_config(self) -> EvaluationConfig:
        """Convert to EvaluationConfig for compatibility."""
        return EvaluationConfig(
            method=self.method,
            mode=self.mode,
            use_vague=self.use_vague,
            parser_type=self.parser_type,
            overwrite=self.overwrite,
            skip_model_eval=self.skip_model_eval
        )
    
    def to_experiment_config(self) -> ConfigExperimentConfig:
        """Convert to ConfigExperimentConfig for compatibility."""
        return ConfigExperimentConfig(
            name=f"{self.model_name}_{self.method}_{self.mode}",
            description=f"Pipeline experiment: {self.model_name} - {self.method} - {self.mode}",
            output_dir=self.output_dir or "./results"
        )


@dataclass
class ExperimentResult:
    """Result of an experiment run."""
    
    success: bool
    results_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    model_name: Optional[str] = None
    method: Optional[str] = None
    provider: Optional[str] = None
    mode: Optional[str] = None
    research_goal: Optional[str] = None
    expanded_results_path: Optional[Path] = None


class ModelFactory:
    """Factory for creating model instances based on configuration."""
    
    @staticmethod
    def create_model(
        model_name: str, 
        config: ModelConfig, 
        parser_type: type, 
        mode: str
    ) -> LLModel:
        """Create a model instance based on the model name and configuration.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration
            parser_type: Parser class to use
            mode: Evaluation mode
            
        Returns:
            Initialized model instance
            
        Raises:
            ModelError: If model creation fails
        """
        kwargs = {
            "model_name": model_name,
            "api_key": config.api_key,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "n_trails": config.n_trails,
            "parser": parser_type,
        }
        if config.reasoning_effort not in {None, "default"}:
            from timeline_extraction.models._reasoning import reasoning_options
            reasoning_options(config.provider, model_name, config.reasoning_effort)
            kwargs["reasoning_effort"] = config.reasoning_effort
        try:
            if config.provider in {"antigravity", "codex", "claude"}:
                if config.provider == "antigravity":
                    from timeline_extraction.models.antigravity import Antigravity
                    adapter = Antigravity
                elif config.provider == "codex":
                    from timeline_extraction.models.codex import Codex
                    adapter = Codex
                else:
                    from timeline_extraction.models.claude import Claude
                    adapter = Claude
                return adapter(**kwargs, cli_timeout=config.cli_timeout)
            if config.provider == "openai":
                from timeline_extraction.models.OpenAIClient import OpenAIClient
                return OpenAIClient(**kwargs, base_url=config.base_url)
            if config.provider == "google":
                from timeline_extraction.models.gemini import Gemini
                return Gemini(**kwargs)
            if config.provider == "groq":
                from timeline_extraction.models.llama3 import GroqModel
                return GroqModel(**kwargs)
            if config.provider == "together":
                from timeline_extraction.models.TogetherAIClient import TogetherAIClient
                return TogetherAIClient(**kwargs)
            if config.provider == "huggingface":
                from timeline_extraction.models.HuggingFaceClient import HuggingfaceClient
                return HuggingfaceClient(**kwargs, device=config.device, cache_dir=config.cache_dir)
            raise ValueError(f"Unsupported provider: {config.provider}")
        except Exception as error:
            raise ModelError(f"Failed to create {config.provider} model {model_name}: {error}") from error


class PromptFactory:
    """Factory for creating prompt instances based on configuration."""
    
    @staticmethod
    def create_prompt(
        method: str, 
        parser_type: type, 
        use_vague: bool
    ) -> Prompt:
        """Create a prompt instance based on method and parser type.
        
        Args:
            method: Evaluation method (zero-shot, few-shot)
            parser_type: Parser class type
            use_vague: Whether to use vague relations
            
        Returns:
            Initialized prompt instance
        """
        is_few_shot = method == "few-shot"
        
        if issubclass(parser_type, LabelParser):
            return PairwisePrompt(use_few_shot=is_few_shot, use_vague=use_vague)
        else:
            return MultiEvents(
                use_few_shot=is_few_shot,
                use_vague=use_vague,
                provide_justification=False,
            )


def generate_suffix_name(
    model_name: str, 
    method: str, 
    suffix_path: str, 
    use_vague: bool = False
) -> str:
    """Generate a suffix name for file naming.
    
    Args:
        model_name: Name of the model
        method: Evaluation method
        suffix_path: Additional suffix path
        use_vague: Whether to include vague indicator
        
    Returns:
        Generated suffix name
    """
    suffixes = [model_name.split("/")[1] if "/" in model_name else model_name, method]
    if use_vague:
        suffixes.append("w_vague")
    if suffix_path:
        suffixes.append(suffix_path)
    return "-".join(suffixes)


class TimelineExtractionPipeline:
    """Main pipeline class for timeline extraction experiments."""
    
    def __init__(
        self, 
        config_manager: Optional[ConfigManager] = None,
        data_path: Optional[Path] = None,
        trc_path: Optional[Path] = None,
        tracking_config: Optional[TrackingConfig] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            config_manager: Configuration manager instance
            data_path: Path to data directory
            trc_path: Path to TRC directory
            tracking_config: Explicit tracing settings; otherwise read from the environment
        """
        self.config_manager = config_manager or ConfigManager()
        self.data_path = Path(data_path or self.config_manager.data_config.data_path)
        self.trc_path = Path(trc_path) if trc_path else self.data_path / "TRC"
        self.matres_path = self.data_path / "MATRES"
        self.tracking_config = tracking_config or TrackingConfig.from_env()
        self._tracker = None
    
    def _get_file_paths(
        self,
        config: PipelineExperimentConfig,
        model_config: Optional[ModelConfig] = None,
    ) -> Dict[str, Path]:
        """Isolate cached artifacts by provider, data, context, and generation settings."""
        model_config = model_config or self.config_manager.resolve_model_config(config.model_name)
        raw_text = self._get_raw_text_path(config)
        gold_path = self._get_labeled_data_path(config)
        settings = {
            "experiment": {
                key: value for key, value in asdict(config).items()
                if key not in {"overwrite", "skip_model_eval", "output_dir"}
            },
            "model": {
                key: value for key, value in asdict(model_config).items()
                if key != "api_key"
            },
            "inputs": {
                str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                for path in (raw_text, gold_path)
            },
        }
        fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", config.model_name)
        dataset_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", config.data_name)
        root = Path(config.output_dir or self.config_manager.data_config.results_path)
        artifact_dir = (
            root / model_config.provider / model_slug / config.mode
            / ("full_context" if config.full_context else "minimal_context")
            / dataset_slug / config.method / fingerprint
        )
        return {
            "llm_response": artifact_dir / "responses.jsonl",
            "parsed_response": artifact_dir / "parsed.csv",
            "results": artifact_dir / "results.csv",
            "expanded_results": artifact_dir / "expanded_results.csv",
            "checkpoint": artifact_dir / "checkpoint.jsonl",
            "metrics": artifact_dir / "metrics.csv",
            "raw_text": raw_text,
        }
    
    def _get_raw_text_path(self, config: PipelineExperimentConfig) -> Path:
        """Get the raw text file path based on configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Path to raw text file
        """
        datasets = self.config_manager.data_config.datasets or {}
        dataset = datasets.get(config.data_name, {})
        prepared_file = (dataset.get("prepared_files") or {}).get(config.mode) or dataset.get("prepared_file")
        if prepared_file:
            return Path(prepared_file)
        if config.mode == "comb":
            context_ref = "full_context" if config.full_context else "minimal_context"
            filename = f"{config.mode}_{config.data_name.lower()}_{context_ref}_text_w_relations_prepared.json"
        else:
            filename = f"{config.mode}_{config.data_name.lower()}_text_w_relations_prepared.json"
        
        return self.trc_path / "raw_text" / filename
    
    def _get_labeled_data_path(self, config: PipelineExperimentConfig) -> Path:
        """Get the labeled data path based on configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Path to labeled data file
        """
        dataset = (self.config_manager.data_config.datasets or {}).get(config.data_name, {})
        if dataset.get("gold_file"):
            return Path(dataset["gold_file"])
        dataset_mapping = {
            "te3-platinum": "platinum.txt",
            "timebank": "timebank.txt", 
            "aquaint": "aquaint.txt"
        }
        
        if config.data_name not in dataset_mapping:
            raise DataError(f"Dataset {config.data_name!r} requires a gold_file override")
        filename = dataset_mapping[config.data_name]
        return self.matres_path / filename
    
    def run_experiment(
        self,
        config: PipelineExperimentConfig,
        model_config: Optional[ModelConfig] = None,
    ) -> ExperimentResult:
        """Run locally, optionally exporting an experiment trace and its generations."""
        if self.tracking_config.backend == "none":
            return self._run_experiment(config, model_config)
        if self._tracker is None:
            self._tracker = ExperimentTracker(self.tracking_config)
        tracker = self._tracker
        try:
            model_config = model_config or self.config_manager.resolve_model_config(config.model_name)
            metadata = {key: value for key, value in asdict(model_config).items() if key != "api_key"}
            with tracker.span("timeline.experiment", asdict(config), metadata) as span:
                result = self._run_experiment(config, model_config, tracker)
                output = {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "artifacts": {
                        name: str(path) for name in (
                            "results_path", "expanded_results_path", "metrics_path", "metadata_path"
                        )
                        if (path := getattr(result, name)) is not None
                    },
                }
                if result.success:
                    metrics = pd.read_csv(result.metrics_path, header=[0, 1])
                    metrics.columns = [
                        name if metric.startswith("Unnamed:") else f"{name}.{metric}"
                        for name, metric in metrics.columns
                    ]
                    output["metrics"] = json.loads(metrics.to_json(orient="records"))
                span.finish(output, error=result.error)
                return result
        finally:
            tracker.flush()

    def _run_experiment(
        self, 
        config: PipelineExperimentConfig,
        model_config: Optional[ModelConfig] = None,
        tracker: Optional[ExperimentTracker] = None,
    ) -> ExperimentResult:
        """Run a single experiment.
        
        Args:
            config: Experiment configuration
            model_config: Model configuration (optional)
            
        Returns:
            Experiment result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting experiment: {config.model_name} - {config.method} - {config.mode}")
            
            model_config = model_config or self.config_manager.resolve_model_config(config.model_name)
            paths = self._get_file_paths(config, model_config)
            for path_name, path in paths.items():
                if path_name != "raw_text":
                    path.parent.mkdir(parents=True, exist_ok=True)

            parser_type = JsonParser if config.parser_type == "json" else LabelParser
            model = None
            if not config.skip_model_eval:
                model = ModelFactory.create_model(
                    config.model_name, model_config, parser_type, config.mode
                )
            prompt = PromptFactory.create_prompt(
                config.method,
                parser_type,
                config.use_vague
            )
            
            # Run the main pipeline
            results_path = self._run_pipeline(
                config, model, prompt, paths, model_config.n_trails, tracker, model_config
            )
            
            # Get summary results
            metrics_path = self._get_summary_results(
                config, results_path, paths
            )
            expanded_results_path = paths["expanded_results"] if config.mode == "multi" else None
            
            # Create metadata
            metadata_path = self._save_metadata(
                config, results_path, metrics_path, start_time, model_config, expanded_results_path
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Experiment completed successfully in {execution_time:.2f}s")
            
            return ExperimentResult(
                success=True,
                results_path=results_path,
                metrics_path=metrics_path,
                metadata_path=metadata_path,
                expanded_results_path=expanded_results_path,
                execution_time=execution_time,
                model_name=config.model_name,
                method=config.method,
                provider=model_config.provider,
                mode=config.mode,
                research_goal=config.research_goal,
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Experiment failed: {str(e)}"
            logger.error(error_msg)
            
            return ExperimentResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                model_name=config.model_name,
                method=config.method,
                provider=model_config.provider if model_config else None,
                mode=config.mode,
                research_goal=config.research_goal,
            )
        finally:
            # Cleanup
            self._cleanup_resources()
    
    def _run_pipeline(
        self,
        config: PipelineExperimentConfig,
        model: Optional[LLModel],
        prompt: Prompt,
        paths: Dict[str, Path],
        n_trails: int = 5,
        tracker: Optional[ExperimentTracker] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> Path:
        """Run the main pipeline processing.
        
        Args:
            config: Experiment configuration
            model: Model instance
            prompt: Prompt instance
            paths: File paths dictionary
            
        Returns:
            Path to results file
        """
        if not config.skip_model_eval:
            logger.info("Generating model responses...")
            model.generate_responses(
                text_path=paths["raw_text"],
                results_path=paths["llm_response"],
                prompt_params=config.prompt_params,
                prompt_template=prompt,
                overwrite=config.overwrite,
                checkpoint_results=paths["checkpoint"],
                response_generator=(
                    lambda prompt: tracker.generate(model, prompt, {
                        "ls_provider": model_config.provider,
                        "ls_model_name": config.model_name,
                        "temperature": model_config.temperature,
                        "max_tokens": model_config.max_tokens,
                        "n_trails": model_config.n_trails,
                    })
                ) if tracker else None,
            )
        elif not paths["llm_response"].is_file():
            raise DataError(f"No saved responses to evaluate: {paths['llm_response']}")

        logger.info("Parsing model responses...")
        self._parse_responses(config, paths)
        logger.info("Applying majority voting...")
        majority_vote_decision_new(
            paths["parsed_response"], paths["results"], n_trails // 2 + 1
        )
        
        return paths["results"]
    
    def _parse_responses(
        self,
        config: PipelineExperimentConfig,
        paths: Dict[str, Path],
    ) -> None:
        """Parse model responses into structured format.
        
        Args:
            config: Experiment configuration
            model: Model instance
            prompt: Prompt instance
            paths: File paths dictionary
        """
        results_df = pd.read_json(paths["llm_response"], lines=True)
        data = []
        for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Parsing responses"):
            if not isinstance(row.response, str):
                continue
            p_label = row.response.strip().upper()
            for e1, e2 in row.relations:
                # Use the same numeric orientation as JsonParser and the gold data.
                numbers = [JsonParser._event_number(event) for event in (e1, e2)]
                if None in numbers or numbers[0] == numbers[1]:
                    continue
                label = p_label
                true_label = str(row.get("true_label", "NO_TRUE_LABEL")).upper()
                if numbers[0] > numbers[1]:
                    e1, e2 = e2, e1
                    label = {"BEFORE": "AFTER", "AFTER": "BEFORE"}.get(label, label)
                    true_label = {"BEFORE": "AFTER", "AFTER": "BEFORE"}.get(true_label, true_label)
                data.append({
                    "docid": row.doc_id,
                    "eiid1": e1,
                    "eiid2": e2,
                    "relation": true_label,
                    "unique_id": "-".join(sorted([e1, e2])),
                    "p_label": label,
                    "mode": config.mode,
                    "model_name": config.model_name,
                    "iter": row.trail,
                    "prompt": row.prompt,
                    "raw_response": row.get("content", row.response),
                })
        
        # Save parsed responses
        parsed_df = pd.DataFrame(data, columns=[
            "docid", "eiid1", "eiid2", "relation", "unique_id", "p_label",
            "mode", "model_name", "iter", "prompt", "raw_response",
        ])
        parsed_df.to_csv(paths["parsed_response"], index=False)
    
    def _get_summary_results(
        self,
        config: PipelineExperimentConfig,
        results_path: Path,
        paths: Dict[str, Path]
    ) -> Path:
        """Get summary results and save metrics.
        
        Args:
            config: Experiment configuration
            results_path: Path to results file
            paths: File paths dictionary
            
        Returns:
            Path to metrics file
        """
        labeled_path = self._get_labeled_data_path(config)
        
        # Load gold data
        gold_df = load_data(labeled_path)
        if not config.use_vague:
            gold_df = gold_df[gold_df["label"] != "VAGUE"]

        # Load results
        results_df = pd.read_csv(results_path)
        results_df = results_df.drop_duplicates(["docid", "unique_id"])

        # Complete the voted graph only; explicit artifacts and generation caches stay untouched.
        summaries = {"explicit": summary_results(results_df, gold_df, config.model_name)}
        if config.mode == "multi":
            expanded_df = Graph(relation_key="p_label").generate_implicit_relations(results_df)
            expanded_df.to_csv(paths["expanded_results"], index=False)
            summaries["expanded"] = summary_results(expanded_df, gold_df, config.model_name)
        for view, summary in summaries.items():
            summary.columns = pd.MultiIndex.from_tuples(
                [(f"{view}.{name}", metric) for name, metric in summary.columns],
                names=summary.columns.names,
            )
        df = pd.concat(summaries.values(), axis=1)
        df["method"] = config.method
        df["suffix_path"] = config.suffix_path
        df["model_name"] = config.model_name
        
        # Save metrics
        metrics_path = paths["metrics"]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(metrics_path, index=False)
        
        return metrics_path
    
    def _save_metadata(
        self,
        config: PipelineExperimentConfig,
        results_path: Path,
        metrics_path: Path,
        start_time: datetime,
        model_config: ModelConfig,
        expanded_results_path: Optional[Path] = None,
    ) -> Path:
        """Save experiment metadata.
        
        Args:
            config: Experiment configuration
            results_path: Path to results file
            metrics_path: Path to metrics file
            start_time: Experiment start time
            
        Returns:
            Path to metadata file
        """
        metadata = {
            "experiment_config": asdict(config),
            "model_config": {
                key: value for key, value in asdict(model_config).items() if key != "api_key"
            },
            "file_paths": {
                "results_path": str(results_path),
                "metrics_path": str(metrics_path),
                "expanded_results_path": str(expanded_results_path) if expanded_results_path else None,
            },
            "timing": {
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            },
            "system_info": {
                "python_version": os.sys.version,
                "pytorch_version": torch.__version__,
            }
        }
        
        metadata_path = metrics_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def _cleanup_resources(self) -> None:
        """Clean up GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()




class BatchExperimentRunner:
    """Runner for batch experiments across multiple models and methods."""
    
    def __init__(self, pipeline: Optional[TimelineExtractionPipeline] = None):
        """Initialize the batch runner.
        
        Args:
            pipeline: Pipeline instance (optional)
        """
        self.pipeline = pipeline or TimelineExtractionPipeline()
    
    def run_batch_experiments(
        self,
        model_names: List[str],
        methods: List[str],
        base_config: PipelineExperimentConfig,
        model_configs: Optional[Dict[str, ModelConfig]] = None
    ) -> List[ExperimentResult]:
        """Run batch experiments across multiple models and methods.
        
        Args:
            model_names: List of model names to evaluate
            methods: List of methods to use
            base_config: Base experiment configuration
            model_configs: Optional model-specific configurations
            
        Returns:
            List of experiment results
        """
        results = []
        total_experiments = len(model_names) * len(methods)
        
        logger.info(f"Starting batch experiments: {total_experiments} total")
        
        for i, model_name in enumerate(model_names):
            for j, method in enumerate(methods):
                experiment_num = i * len(methods) + j + 1
                
                logger.info(f"Running experiment {experiment_num}/{total_experiments}: {model_name} - {method}")
                
                # Create experiment config
                config = replace(base_config, model_name=model_name, method=method)
                
                # Get model config if available
                model_config = model_configs.get(model_name) if model_configs else None
                
                # Run experiment
                result = self.pipeline.run_experiment(config, model_config)
                results.append(result)
                
                if result.success:
                    logger.info(f"✓ Completed {model_name} - {method}")
                else:
                    logger.error(f"✗ Failed {model_name} - {method}: {result.error}")
                
                logger.info("-" * 50)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch experiments completed: {successful}/{total_experiments} successful")
        
        return results
    
    def save_batch_summary(
        self,
        results: List[ExperimentResult],
        output_path: Path
    ) -> None:
        """Save a summary of batch experiment results.
        
        Args:
            results: List of experiment results
            output_path: Path to save summary
        """
        summary_data = []
        
        for result in results:
            summary_data.append({
                "model_name": result.model_name,
                "method": result.method,
                "provider": result.provider,
                "mode": result.mode,
                "research_goal": result.research_goal,
                "success": result.success,
                "execution_time": result.execution_time,
                "error": result.error,
                "results_path": str(result.results_path) if result.results_path else None,
                "expanded_results_path": str(result.expanded_results_path) if result.expanded_results_path else None,
                "metrics_path": str(result.metrics_path) if result.metrics_path else None,
            })
        
        summary_df = pd.DataFrame(summary_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Batch summary saved to {output_path}")






