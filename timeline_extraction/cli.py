import click
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import os
import torch
import gc
import json
import yaml
from datetime import datetime

from timeline_extraction.pipeline import (
    TimelineExtractionPipeline,
    BatchExperimentRunner,
    PipelineExperimentConfig,
    ModelFactory,
    PromptFactory
)
from timeline_extraction.config import (
    ConfigManager,
    ModelConfig,
    EvaluationConfig,
    create_default_config
)
from timeline_extraction.data.preprocessing import load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("./data")
MATRES_DATA_PATH = DATA_PATH / "MATRES"
TRC_RAW_PATH = DATA_PATH / "TRC"


def setup_config_manager(config_file: Optional[str], ctx_config: dict) -> ConfigManager:
    """Load and setup configuration manager from file and context."""
    config_manager = ConfigManager()
    
    # Load from file if provided
    if config_file:
        config_manager.load_config(config_file)
    
    # Merge context config if provided
    if ctx_config:
        # This would need to be implemented in ConfigManager if needed
        pass
    
    return config_manager


def setup_gpu_device(gpu_device: int) -> None:
    """Set up GPU device environment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)


def create_pipeline_experiment_config(
    model_name: str,
    method: str,
    mode: str,
    data_name: str,
    use_vague: bool,
    parser_type: str,
    overwrite: bool,
    skip_model_eval: bool,
    full_context: bool,
    gpu_device: int,
    output_dir: str,
    **kwargs
) -> PipelineExperimentConfig:
    """Create a pipeline experiment configuration from CLI parameters."""
    return PipelineExperimentConfig(
        model_name=model_name,
        method=method,
        mode=mode,
        data_name=data_name,
        use_vague=use_vague,
        parser_type=parser_type,
        overwrite=overwrite,
        skip_model_eval=skip_model_eval,
        full_context=full_context,
        gpu_device=gpu_device,
        output_dir=output_dir,
        **kwargs
    )


def run_single_evaluation(
    model_name: str,
    method: str,
    mode: str,
    data_name: str,
    gpu_device: int,
    use_vague: bool,
    parser_type: str,
    overwrite: bool,
    skip_model_eval: bool,
    output_dir: str,
    full_context: bool,
    config_manager: ConfigManager,
) -> tuple:
    """Run a single model evaluation using the new pipeline and return results."""
    try:
        # Create pipeline experiment configuration
        experiment_config = create_pipeline_experiment_config(
            model_name=model_name,
            method=method,
            mode=mode,
            data_name=data_name,
            use_vague=use_vague,
            parser_type=parser_type,
            overwrite=overwrite,
            skip_model_eval=skip_model_eval,
            full_context=full_context,
            gpu_device=gpu_device,
            output_dir=output_dir,
            suffix_path="completion",
            prompt_params=["text", "relations"]
        )

        # Get model configuration if available
        model_config = None
        if hasattr(config_manager, 'model_configs'):
            # Try to find model config by provider
            for provider, config in config_manager.model_configs.items():
                if provider.lower() in model_name.lower() or model_name.lower() in provider.lower():
                    model_config = config
                    break

        # Initialize pipeline
        pipeline = TimelineExtractionPipeline(
            config_manager=config_manager,
            data_path=DATA_PATH,
            trc_path=TRC_RAW_PATH
        )

        # Run experiment
        result = pipeline.run_experiment(experiment_config, model_config)

        if result.success:
            return result.metrics_path, result.metadata_path, None
        else:
            return None, None, result.error

    except Exception as e:
        logger.error(f"Error in single evaluation: {e}")
        return None, None, str(e)


def cleanup_model_resources():
    """Clean up GPU memory and run garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()


def add_common_evaluation_options(func):
    """Decorator to add common evaluation options to commands."""
    func = click.option(
        "--mode",
        type=click.Choice(["pair", "multi", "comb"]),
        required=True,
        help="Evaluation mode to use",
    )(func)
    func = click.option(
        "--data-name",
        type=click.Choice(["te3-platinum", "timebank", "aquaint"]),
        required=True,
        help="Name of the dataset to evaluate on",
    )(func)
    func = click.option(
        "--gpu-device", type=int, default=0, help="GPU device to use (default: 0)"
    )(func)
    func = click.option(
        "--use-vague/--no-use-vague",
        default=True,
        help="Whether to include VAGUE relations in evaluation",
    )(func)
    func = click.option(
        "--parser-type",
        type=click.Choice(["label", "json"]),
        default="json",
        help="Type of parser to use for model responses",
    )(func)
    func = click.option(
        "--overwrite/--no-overwrite",
        default=False,
        help="Whether to overwrite existing results",
    )(func)
    func = click.option(
        "--skip-model-eval/--no-skip-model-eval",
        default=False,
        help="Whether to skip model evaluation and only process existing results",
    )(func)
    func = click.option(
        "--output-dir",
        type=click.Path(),
        default="results",
        help="Output directory for results",
    )(func)
    func = click.option(
        "--full-context/--minimal-context",
        default=True,
        help="Use full context for comb mode (default: True)",
    )(func)
    func = click.option(
        "--config-file",
        type=click.Path(exists=True),
        help="Configuration file for model parameters",
    )(func)
    return func


@click.group()
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Set logging level",
)
@click.pass_context
def cli(ctx, config, verbose, log_level):
    """Timeline Extraction: Elegant Temporal Relation Classification System

    A comprehensive CLI tool for evaluating temporal relation classification
    using Large Language Models with the new elegant pipeline architecture.
    
    Features:
    - Clean, object-oriented pipeline design
    - Multi-model support (OpenAI, Google, HuggingFace, etc.)
    - Batch experiment processing
    - Configuration management
    - Comprehensive logging and error handling
    
    Commands:
    - model evaluate: Run single model evaluation
    - model batch-evaluate: Run batch evaluations across multiple models
    - demo-pipeline: Demonstrate the new elegant pipeline
    - generate-config: Generate default configuration file
    - analyze-data: Analyze dataset characteristics
    """
    # Set up logging
    if verbose:
        log_level = "DEBUG"

    logging.getLogger().setLevel(getattr(logging, log_level))

    # Load configuration if provided
    ctx.ensure_object(dict)
    if config:
        with open(config, "r") as f:
            if config.endswith(".yaml") or config.endswith(".yml"):
                ctx.obj["config"] = yaml.safe_load(f)
            else:
                ctx.obj["config"] = json.load(f)
    else:
        ctx.obj["config"] = {}


# Data analysis command
@cli.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the data file to analyze",
)
@click.option("--output", type=click.Path(), help="Output file for analysis results")
def analyze_data(data_path: str, output: Optional[str]):
    """Analyze dataset statistics and characteristics."""
    logger.info(f"Analyzing data from {data_path}")

    try:
        # Load and analyze data
        data = load_data(Path(data_path))
        logger.info(f"Loaded {len(data)} documents")

        # Basic analysis
        analysis_results = {
            "document_count": len(data),
            "columns": list(data.columns) if hasattr(data, "columns") else None,
            "sample_size": min(10, len(data)) if len(data) > 0 else 0,
        }

        if output:
            with open(output, "w") as f:
                json.dump(analysis_results, f, indent=2)
            logger.info(f"Analysis results saved to {output}")

        logger.info("Data analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        raise


# Model evaluation commands
@cli.group()
def model():
    """Model evaluation and management commands."""
    pass


@model.command()
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="Name of the model to use (e.g., gpt-4o-mini, gemini-1.5-pro, meta-llama/Llama-3.1-8B-Instruct)",
)
@click.option(
    "--method",
    type=click.Choice(["zero-shot", "few-shot"]),
    required=True,
    help="Evaluation method to use",
)
@add_common_evaluation_options
@click.pass_context
def evaluate(
    ctx,
    model_name: str,
    method: str,
    mode: str,
    data_name: str,
    gpu_device: int,
    use_vague: bool,
    parser_type: str,
    overwrite: bool,
    skip_model_eval: bool,
    output_dir: str,
    full_context: bool,
    config_file: Optional[str],
):
    """Evaluate the model on test data using the new elegant pipeline."""
    logger.info(
        f"Starting evaluation with model {model_name} using {method} method in {mode} mode"
    )

    # Setup configuration and GPU
    config_manager = setup_config_manager(config_file, ctx.obj.get("config", {}))
    setup_gpu_device(gpu_device)

    # Run single evaluation using new pipeline
    results_metrics_path, metadata_path, error = run_single_evaluation(
        model_name=model_name,
        method=method,
        mode=mode,
        data_name=data_name,
        gpu_device=gpu_device,
        use_vague=use_vague,
        parser_type=parser_type,
        overwrite=overwrite,
        skip_model_eval=skip_model_eval,
        output_dir=output_dir,
        full_context=full_context,
        config_manager=config_manager,
    )

    if error:
        logger.error(f"Error during evaluation: {error}")
        raise Exception(error)

    logger.info(f"Evaluation completed. Results saved to {results_metrics_path}")
    logger.info(f"Metadata saved to {metadata_path}")


@model.command()
@click.option(
    "--model-names",
    type=str,
    multiple=True,
    required=True,
    help="Names of models to evaluate (can be specified multiple times)",
)
@click.option(
    "--methods",
    type=click.Choice(["zero-shot", "few-shot"]),
    multiple=True,
    default=["zero-shot", "few-shot"],
    help="Evaluation methods to use",
)
@add_common_evaluation_options
@click.pass_context
def batch_evaluate(
    ctx,
    model_names: List[str],
    methods: List[str],
    mode: str,
    data_name: str,
    gpu_device: int,
    use_vague: bool,
    parser_type: str,
    overwrite: bool,
    skip_model_eval: bool,
    output_dir: str,
    full_context: bool,
    config_file: Optional[str],
):
    """Run batch evaluation across multiple models and methods using the new pipeline."""
    logger.info(
        f"Starting batch evaluation with {len(model_names)} models and {len(methods)} methods"
    )

    # Setup configuration and GPU
    config_manager = setup_config_manager(config_file, ctx.obj.get("config", {}))
    setup_gpu_device(gpu_device)

    try:
        # Create base experiment configuration
        base_config = create_pipeline_experiment_config(
            model_name="",  # Will be set per experiment
            method="",      # Will be set per experiment
            mode=mode,
            data_name=data_name,
            use_vague=use_vague,
            parser_type=parser_type,
            overwrite=overwrite,
            skip_model_eval=skip_model_eval,
            full_context=full_context,
            gpu_device=gpu_device,
            output_dir=output_dir,
            suffix_path="completion",
            prompt_params=["text", "relations"]
        )

        # Prepare model configurations
        model_configs = {}
        if hasattr(config_manager, 'model_configs'):
            for provider, config in config_manager.model_configs.items():
                # Try to match model names to providers
                for model_name in model_names:
                    if provider.lower() in model_name.lower() or model_name.lower() in provider.lower():
                        model_configs[model_name] = config

        # Initialize pipeline and batch runner
        pipeline = TimelineExtractionPipeline(
            config_manager=config_manager,
            data_path=DATA_PATH,
            trc_path=TRC_RAW_PATH
        )
        batch_runner = BatchExperimentRunner(pipeline)

        # Run batch experiments
        results = batch_runner.run_batch_experiments(
            model_names=list(model_names),
            methods=list(methods),
            base_config=base_config,
            model_configs=model_configs
        )

        # Save batch summary
        summary_path = Path(output_dir) / "batch_summary.csv"
        batch_runner.save_batch_summary(results, summary_path)

        # Print summary
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        logger.info(f"Batch evaluation completed. {successful}/{total} evaluations finished successfully.")
        logger.info(f"Batch summary saved to {summary_path}")

    except Exception as e:
        logger.error(f"Error during batch evaluation: {e}")
        raise


# Configuration management
@cli.command()
@click.option(
    "--output",
    type=click.Path(),
    default="config.yaml",
    help="Output configuration file path",
)
def generate_config(output: str):
    """Generate a default configuration file."""
    logger.info(f"Generating configuration file: {output}")

    try:
        default_config = create_default_config()

        with open(output, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration file generated: {output}")
    except Exception as e:
        logger.error(f"Failed to generate configuration file: {e}")
        raise


# Pipeline demonstration command
@cli.command()
@click.option(
    "--model-names",
    type=str,
    multiple=True,
    default=["gpt-4o-mini"],
    help="Names of models to evaluate (can be specified multiple times)",
)
@click.option(
    "--methods",
    type=click.Choice(["zero-shot", "few-shot"]),
    multiple=True,
    default=["zero-shot", "few-shot"],
    help="Evaluation methods to use",
)
@click.option(
    "--mode",
    type=click.Choice(["pair", "multi", "comb"]),
    default="comb",
    help="Evaluation mode to use",
)
@click.option(
    "--data-name",
    type=click.Choice(["te3-platinum", "timebank", "aquaint"]),
    default="te3-platinum",
    help="Name of the dataset to evaluate on",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./results/demo",
    help="Output directory for results",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Configuration file for model parameters",
)
@click.pass_context
def demo_pipeline(
    ctx,
    model_names: List[str],
    methods: List[str],
    mode: str,
    data_name: str,
    output_dir: str,
    config_file: Optional[str],
):
    """Demonstrate the new elegant pipeline with a simple example."""
    logger.info("Running pipeline demonstration...")
    
    # Setup configuration and GPU
    config_manager = setup_config_manager(config_file, ctx.obj.get("config", {}))
    setup_gpu_device(0)  # Use GPU 0 for demo
    
    try:
        # Create base experiment configuration
        base_config = create_pipeline_experiment_config(
            model_name="",  # Will be set per experiment
            method="",      # Will be set per experiment
            mode=mode,
            data_name=data_name,
            use_vague=True,
            parser_type="json",
            overwrite=True,
            skip_model_eval=False,
            full_context=True,
            gpu_device=0,
            output_dir=output_dir,
            suffix_path="demo",
            prompt_params=["text", "relations"]
        )

        # Initialize pipeline and batch runner
        pipeline = TimelineExtractionPipeline(
            config_manager=config_manager,
            data_path=DATA_PATH,
            trc_path=TRC_RAW_PATH
        )
        batch_runner = BatchExperimentRunner(pipeline)

        # Run batch experiments
        results = batch_runner.run_batch_experiments(
            model_names=list(model_names),
            methods=list(methods),
            base_config=base_config
        )

        # Save batch summary
        summary_path = Path(output_dir) / "demo_summary.csv"
        batch_runner.save_batch_summary(results, summary_path)

        # Print results
        print("\n" + "="*60)
        print("PIPELINE DEMONSTRATION RESULTS")
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
        
        print(f"\nDemo summary saved to: {summary_path}")
        logger.info("Pipeline demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Error during pipeline demonstration: {e}")
        raise


if __name__ == "__main__":
    cli()
