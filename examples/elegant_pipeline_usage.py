#!/usr/bin/env python3
"""
Example usage of the elegant Timeline Extraction Pipeline.

This example demonstrates how to use the new clean and elegant pipeline
for running temporal relation extraction experiments with multiple models.
"""

import os
import logging
from pathlib import Path

from timeline_extraction.pipeline import (
    TimelineExtractionPipeline,
    BatchExperimentRunner,
    PipelineExperimentConfig
)
from timeline_extraction.config import ModelConfig, ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def single_experiment_example():
    """Example of running a single experiment."""
    logger.info("Running single experiment example...")
    
    # Set up GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Create experiment configuration
    config = PipelineExperimentConfig(
        model_name="gpt-4o-mini",
        method="zero-shot",
        mode="comb",  # all possible pairs in 2-line apart context
        data_name="te3-platinum",
        use_vague=True,
        parser_type="json",
        overwrite=True,
        skip_model_eval=False,
        full_context=True,
        suffix_path="completion",
        prompt_params=["text", "relations"],
        gpu_device=0,
        output_dir="./results/single_experiment"
    )
    
    # Initialize pipeline
    pipeline = TimelineExtractionPipeline()
    
    # Run experiment
    result = pipeline.run_experiment(config)
    
    # Check results
    if result.success:
        logger.info(f"✓ Experiment completed successfully!")
        logger.info(f"  Execution time: {result.execution_time:.2f}s")
        logger.info(f"  Results: {result.results_path}")
        logger.info(f"  Metrics: {result.metrics_path}")
        logger.info(f"  Metadata: {result.metadata_path}")
    else:
        logger.error(f"✗ Experiment failed: {result.error}")
    
    return result


def batch_experiment_example():
    """Example of running batch experiments across multiple models and methods."""
    logger.info("Running batch experiment example...")
    
    # Set up GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Define models and methods to test
    model_names = [
        "gpt-4o-mini",
        # "gpt-4o",  # Uncomment to test more models
        # "gemini-1.5-pro",
        # "meta-llama/Llama-3.1-8B-Instruct"
    ]
    
    methods = ["zero-shot", "few-shot"]
    
    # Base configuration for all experiments
    base_config = PipelineExperimentConfig(
        model_name="",  # Will be set per experiment
        method="",      # Will be set per experiment
        mode="comb",
        data_name="te3-platinum",
        use_vague=True,
        parser_type="json",
        overwrite=True,
        skip_model_eval=False,
        full_context=True,
        suffix_path="completion",
        prompt_params=["text", "relations"],
        gpu_device=0,
        output_dir="./results/batch_experiments"
    )
    
    # Optional: Define model-specific configurations
    model_configs = {
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            max_tokens=2000,
            temperature=0.0
        ),
        # Add more model configs as needed
    }
    
    # Initialize pipeline and batch runner
    pipeline = TimelineExtractionPipeline()
    batch_runner = BatchExperimentRunner(pipeline)
    
    # Run batch experiments
    results = batch_runner.run_batch_experiments(
        model_names=model_names,
        methods=methods,
        base_config=base_config,
        model_configs=model_configs
    )
    
    # Save batch summary
    summary_path = Path("./results/batch_experiments/batch_summary.csv")
    batch_runner.save_batch_summary(results, summary_path)
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    total = len(results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH EXPERIMENT RESULTS: {successful}/{total} successful")
    logger.info(f"{'='*60}")
    
    for result in results:
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        logger.info(f"{result.model_name} - {result.method}: {status}")
        if result.success:
            logger.info(f"  Execution time: {result.execution_time:.2f}s")
        else:
            logger.info(f"  Error: {result.error}")
    
    logger.info(f"\nBatch summary saved to: {summary_path}")
    
    return results


def configuration_example():
    """Example of using configuration files."""
    logger.info("Running configuration example...")
    
    # Create a configuration manager
    config_manager = ConfigManager()
    
    # You can load from a config file if it exists
    # config_manager.load_config("config.yaml")
    
    # Or create a default configuration
    from timeline_extraction.config import create_default_config
    default_config = create_default_config()
    
    # Save default config for reference
    config_path = Path("./config_example.yaml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Example configuration saved to: {config_path}")
    
    # Initialize pipeline with config manager
    pipeline = TimelineExtractionPipeline(config_manager=config_manager)
    
    return pipeline


def main():
    """Main function to run all examples."""
    logger.info("Starting Timeline Extraction Pipeline Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Single experiment
        logger.info("\n1. Single Experiment Example")
        logger.info("-" * 30)
        single_result = single_experiment_example()
        
        # Example 2: Batch experiments
        logger.info("\n2. Batch Experiment Example")
        logger.info("-" * 30)
        batch_results = batch_experiment_example()
        
        # Example 3: Configuration management
        logger.info("\n3. Configuration Example")
        logger.info("-" * 30)
        config_pipeline = configuration_example()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
