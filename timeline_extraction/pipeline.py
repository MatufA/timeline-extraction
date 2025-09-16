"""Timeline Extraction Pipeline - Elegant and Clean Implementation

This module provides a clean, configurable pipeline for temporal relation extraction
using Large Language Models with support for multiple models and evaluation methods.
"""

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
import torch
from tqdm.auto import tqdm

from timeline_extraction.data.postprocessing import majority_vote_decision_new
from timeline_extraction.data.preprocessing import load_data
from timeline_extraction.metrics import summary_results
from timeline_extraction.models.LLModel import LLModel, LabelParser, JsonParser
from timeline_extraction.prompts.Prompt import Prompt, PairwisePrompt, MultiEvents
from timeline_extraction.config import ConfigManager, ModelConfig, EvaluationConfig, ExperimentConfig as ConfigExperimentConfig
from timeline_extraction.exceptions import ModelError, DataError, EvaluationError

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_PATH = Path("./data")
DEFAULT_MATRES_PATH = DEFAULT_DATA_PATH / "MATRES"
DEFAULT_TRC_PATH = DEFAULT_DATA_PATH / "TRC"


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
    parser_type: str = "json"  # json, label
    overwrite: bool = False
    skip_model_eval: bool = False
    
    # Pipeline-specific settings
    full_context: bool = True
    suffix_path: str = "completion"
    prompt_params: List[str] = field(default_factory=lambda: ["text", "relations"])
    gpu_device: int = 0
    output_dir: str = "./results"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.method not in ["zero-shot", "few-shot"]:
            raise ValueError(f"Invalid method: {self.method}")
        if self.mode not in ["pair", "multi", "comb"]:
            raise ValueError(f"Invalid mode: {self.mode}")
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
            output_dir=self.output_dir
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
        try:
            if "gpt" in model_name.lower():
                from timeline_extraction.models.OpenAIClient import OpenAIClient
                return OpenAIClient(
                    model_name=model_name,
                    use_formate=False,
                    parser=parser_type,
                    use_dot_graph=(mode == "multi"),
                )
            elif "gemini" in model_name.lower():
                from timeline_extraction.models.gemini import Gemini
                return Gemini(model_name=model_name, n_trails=5)
            elif "llama" in model_name.lower() and "groq" in model_name.lower():
                from timeline_extraction.models.llama3 import GroqModel
                return GroqModel(model_name=model_name)
            elif "meta-llama" in model_name.lower() or "mistralai" in model_name.lower():
                from timeline_extraction.models.TogetherAIClient import TogetherAIClient
                return TogetherAIClient(model_name=model_name)
            else:
                # Default to HuggingFace for other models
                from timeline_extraction.models.HuggingFaceClient import HuggingfaceClient
                return HuggingfaceClient(
                    model_name=model_name, 
                    device=config.device, 
                    parser=parser_type
                )
        except Exception as e:
            raise ModelError(f"Failed to create model {model_name}: {str(e)}")


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
        trc_path: Optional[Path] = None
    ):
        """Initialize the pipeline.
        
        Args:
            config_manager: Configuration manager instance
            data_path: Path to data directory
            trc_path: Path to TRC directory
        """
        self.config_manager = config_manager or ConfigManager()
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.trc_path = trc_path or DEFAULT_TRC_PATH
        self.matres_path = self.data_path / "MATRES"
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.trc_path / "llm_response",
            self.trc_path / "parsed_responses", 
            self.trc_path / "results",
            self.trc_path / "llm_raw_response",
            self.trc_path / "final_metrics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_file_paths(
        self, 
        config: PipelineExperimentConfig
    ) -> Dict[str, Path]:
        """Generate file paths for the experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary of file paths
        """
        suffix_name = generate_suffix_name(
            config.model_name, 
            config.method, 
            config.suffix_path, 
            config.use_vague
        )
        
        base_name = f"{config.mode}-{config.data_name}-{suffix_name}"
        
        return {
            "llm_response": self.trc_path / "llm_response" / config.method / f"{base_name}.jsonl",
            "parsed_response": self.trc_path / "parsed_responses" / config.method / f"{base_name}-results.csv",
            "results": self.trc_path / "results" / config.method / f"{base_name}-results.csv",
            "checkpoint": self.trc_path / "llm_raw_response" / config.method / f"{base_name}.jsonl",
            "raw_text": self._get_raw_text_path(config)
        }
    
    def _get_raw_text_path(self, config: PipelineExperimentConfig) -> Path:
        """Get the raw text file path based on configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Path to raw text file
        """
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
        dataset_mapping = {
            "te3-platinum": "platinum.txt",
            "timebank": "timebank.txt", 
            "aquaint": "aquaint.txt"
        }
        
        filename = dataset_mapping.get(config.data_name, "platinum.txt")
        return self.matres_path / filename
    
    def run_experiment(
        self, 
        config: PipelineExperimentConfig,
        model_config: Optional[ModelConfig] = None
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
            
            # Get file paths
            paths = self._get_file_paths(config)
            
            # Create model and prompt
            parser_type = JsonParser if config.parser_type == "json" else LabelParser
            model = ModelFactory.create_model(
                config.model_name, 
                model_config or ModelConfig(name=config.model_name),
                parser_type,
                config.mode
            )
            
            prompt = PromptFactory.create_prompt(
                config.method,
                parser_type,
                config.use_vague
            )
            
            # Run the main pipeline
            results_path = self._run_pipeline(
                config, model, prompt, paths
            )
            
            # Get summary results
            metrics_path = self._get_summary_results(
                config, results_path, paths
            )
            
            # Create metadata
            metadata_path = self._save_metadata(
                config, results_path, metrics_path, start_time
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Experiment completed successfully in {execution_time:.2f}s")
            
            return ExperimentResult(
                success=True,
                results_path=results_path,
                metrics_path=metrics_path,
                metadata_path=metadata_path,
                execution_time=execution_time,
                model_name=config.model_name,
                method=config.method
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
                method=config.method
            )
        finally:
            # Cleanup
            self._cleanup_resources()
    
    def _run_pipeline(
        self,
        config: PipelineExperimentConfig,
        model: LLModel,
        prompt: Prompt,
        paths: Dict[str, Path]
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
        # Ensure checkpoint directory exists
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        
        # Generate model responses if not skipping
        if not config.skip_model_eval:
            logger.info("Generating model responses...")
        model.generate_responses(
                text_path=paths["raw_text"],
                results_path=paths["llm_response"],
                prompt_params=config.prompt_params,
            prompt_template=prompt,
                overwrite=config.overwrite,
                checkpoint_results=paths["checkpoint"],
            )
        
        # Parse responses
        logger.info("Parsing model responses...")
        self._parse_responses(config, model, prompt, paths)
        
        # Apply majority voting
        logger.info("Applying majority voting...")
        min_votes = 3
        majority_vote_decision_new(
            paths["parsed_response"], 
            paths["results"], 
            min_votes
        )
        
        return paths["results"]
    
    def _parse_responses(
        self,
        config: PipelineExperimentConfig,
        model: LLModel,
        prompt: Prompt,
        paths: Dict[str, Path]
    ) -> None:
        """Parse model responses into structured format.
        
        Args:
            config: Experiment configuration
            model: Model instance
            prompt: Prompt instance
            paths: File paths dictionary
        """
        # Load results
        results_df = pd.read_json(paths["llm_response"], lines=True)
        
        # Parse each response
        data = []
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Parsing responses"):
            if isinstance(prompt, PairwisePrompt):
                true_labels = (
                    [row.true_label.upper()] if "true_label" in row else ["NO_TRUE_LABEL"]
                )
            else:
                true_labels = ["NO_TRUE_LABEL"]

            if isinstance(row.response, str):
                p_label = row.response.upper()
            else:
                p_label = row.response["relation"].upper()

            for true_label, (e1, e2) in zip(true_labels, row.relations):
                data.append({
                    "docid": row.doc_id,
                    "eiid1": e1,
                    "eiid2": e2,
                    "relation": true_label,
                    "unique_id": "-".join(sorted([e1, e2])),
                    "p_label": p_label,
                    "mode": config.mode,
                    "model_name": config.model_name,
                    "iter": row.trail,
                    "prompt": row.prompt,
                    "raw_response": row.content if "content" in row else row.response,
                })
        
        # Save parsed responses
        parsed_df = pd.DataFrame(data)
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
        results_df = pd.merge(
            results_df,
            gold_df[["docid", "unique_id"]],
            how="inner",
            on=["docid", "unique_id"],
        ).drop_duplicates(["docid", "unique_id"])

        # Calculate summary
        df = summary_results(results_df, gold_df, config.model_name)
        df["method"] = config.method
        df["suffix_path"] = config.suffix_path
        df["model_name"] = config.model_name
        
        # Save metrics
        metrics_path = (
            Path(config.output_dir) / "final_metrics" / config.method / results_path.name
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(metrics_path, index=False)
        
        return metrics_path
    
    def _save_metadata(
        self,
        config: PipelineExperimentConfig,
        results_path: Path,
        metrics_path: Path,
        start_time: datetime
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
            "experiment_config": {
                "model_name": config.model_name,
                "method": config.method,
                "mode": config.mode,
                "data_name": config.data_name,
                "use_vague": config.use_vague,
                "parser_type": config.parser_type,
                "full_context": config.full_context,
                "suffix_path": config.suffix_path,
                "prompt_params": config.prompt_params,
                "gpu_device": config.gpu_device,
                "output_dir": config.output_dir,
            },
            "file_paths": {
                "results_path": str(results_path),
                "metrics_path": str(metrics_path),
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


# Legacy function for backward compatibility
def main(
    model_name: str,
    method: str,
    model: LLModel,
    prompt_params: List[str],
    raw_text_name: str,
    data_name: str,
    prompt: Prompt,
    suffix_path: str = "",
    mode: str = "multi",
    overwrite: bool = False,
    use_vague: bool = False,
    skip_model_eval: bool = False,
) -> Path:
    """Legacy main function for backward compatibility.
    
    This function is deprecated. Use TimelineExtractionPipeline instead.
    """
    logger.warning("Using deprecated main() function. Consider using TimelineExtractionPipeline instead.")
    
    config = PipelineExperimentConfig(
        model_name=model_name,
        method=method,
        mode=mode,
        data_name=data_name,
        use_vague=use_vague,
        overwrite=overwrite,
        skip_model_eval=skip_model_eval,
        suffix_path=suffix_path,
        prompt_params=prompt_params
    )
    
    pipeline = TimelineExtractionPipeline()
    result = pipeline.run_experiment(config)
    
    if not result.success:
        raise PipelineError(f"Pipeline failed: {result.error}")
    
    return result.results_path


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
                config = PipelineExperimentConfig(
                    model_name=model_name,
                    method=method,
                    mode=base_config.mode,
                    data_name=base_config.data_name,
                    use_vague=base_config.use_vague,
                    parser_type=base_config.parser_type,
                    overwrite=base_config.overwrite,
                    skip_model_eval=base_config.skip_model_eval,
                    full_context=base_config.full_context,
                    suffix_path=base_config.suffix_path,
                    prompt_params=base_config.prompt_params,
                    gpu_device=base_config.gpu_device,
                    output_dir=base_config.output_dir,
                )
                
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
                "success": result.success,
                "execution_time": result.execution_time,
                "error": result.error,
                "results_path": str(result.results_path) if result.results_path else None,
                "metrics_path": str(result.metrics_path) if result.metrics_path else None,
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Batch summary saved to {output_path}")


# Legacy function for backward compatibility
def get_summary_results(
    model_name: str,
    method: str,
    labeled_path: Path,
    results_path: Path,
    suffix_path: str = "",
    use_vague: bool = True,
) -> pd.DataFrame:
    """Legacy function for getting summary results.
    
    This function is deprecated. Use TimelineExtractionPipeline instead.
    """
    logger.warning("Using deprecated get_summary_results() function. Consider using TimelineExtractionPipeline instead.")
    
    gold_df = load_data(labeled_path)
    if not use_vague:
        gold_df = gold_df[gold_df["label"] != "VAGUE"]

    results_df = pd.read_csv(results_path)
    results_df = pd.merge(
        results_df,
        gold_df[["docid", "unique_id"]],
        how="inner",
        on=["docid", "unique_id"],
    ).drop_duplicates(["docid", "unique_id"])

    df = summary_results(results_df, gold_df, model_name)
    df["method"] = method
    df["suffix_path"] = suffix_path
    df["model_name"] = model_name
    return df


def run_example_experiment():
    """Example of how to use the new elegant pipeline."""
    
    # Set up GPU device
    gpu_device = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    
    # Configuration
    model_names = ["gpt-4o-mini"]
    methods = ["zero-shot", "few-shot"]
    
    # Base experiment configuration
    base_config = PipelineExperimentConfig(
        model_name="",  # Will be set per experiment
        method="",      # Will be set per experiment
        mode="comb",    # all possible pairs in 2-line apart context
        data_name="te3-platinum",
        use_vague=True,
        parser_type="json",
        overwrite=True,
        skip_model_eval=False,
        full_context=True,
        suffix_path="completion",
        prompt_params=["text", "relations"],
        gpu_device=gpu_device,
        output_dir="./results"
    )
    
    # Initialize pipeline and batch runner
    pipeline = TimelineExtractionPipeline()
    batch_runner = BatchExperimentRunner(pipeline)
    
    # Run batch experiments
    results = batch_runner.run_batch_experiments(
        model_names=model_names,
        methods=methods,
        base_config=base_config
    )
    
    # Save batch summary
    summary_path = Path("./results/batch_summary.csv")
    batch_runner.save_batch_summary(results, summary_path)
    
    # Print results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        print(f"{result.model_name} - {result.method}: {status}")
        if result.success:
            print(f"  Execution time: {result.execution_time:.2f}s")
            print(f"  Results: {result.results_path}")
            print(f"  Metrics: {result.metrics_path}")
        else:
            print(f"  Error: {result.error}")
        print("-" * 40)
    
    print(f"\nBatch summary saved to: {summary_path}")


if __name__ == "__main__":
    # Run the example experiment
    run_example_experiment()
