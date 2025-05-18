import click
from typing import Optional
import logging
from pathlib import Path
import os
import torch
import gc

from full_temporal_relation.pipeline import main, get_summary_results
from full_temporal_relation.models.HuggingFaceClient import HuggingfaceClient
from full_temporal_relation.models.OpenAIClient import OpenAIClient
from full_temporal_relation.models.LLModel import LLModel, LabelParser, JsonParser
from full_temporal_relation.prompts.Prompt import Prompt, PairwisePrompt, MultiEvents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Full Temporal Relation CLI tool."""
    pass

@cli.command()
@click.option('--input-dir', type=click.Path(exists=True), required=True,
              help='Input directory containing the data to prepare')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Output directory for prepared data')
def prepare_data(input_dir: str, output_dir: str):
    """Prepare and preprocess the data for training."""
    logger.info(f"Preparing data from {input_dir} to {output_dir}")
    # TODO: Implement data preparation logic

@cli.command()
@click.option('--model-name', type=str, required=True,
              help='Name of the model to use (e.g., gpt-4, mistralai/Mistral-7B-Instruct-v0.3)')
@click.option('--method', type=click.Choice(['zero-shot', 'few-shot']), required=True,
              help='Evaluation method to use')
@click.option('--mode', type=click.Choice(['pair', 'multi', 'comb']), required=True,
              help='Evaluation mode to use')
@click.option('--data-name', type=str, required=True,
              help='Name of the dataset to evaluate on (e.g., te3-platinum)')
@click.option('--labeled-path', type=click.Path(exists=True), required=True,
              help='Path to the labeled data file')
@click.option('--raw-text-name', type=str, required=True,
              help='Name of the raw text file to evaluate on')
@click.option('--gpu-device', type=int, default=0,
              help='GPU device to use (default: 0)')
@click.option('--use-vague/--no-use-vague', default=True,
              help='Whether to include VAGUE relations in evaluation')
@click.option('--parser-type', type=click.Choice(['label', 'json']), default='json',
              help='Type of parser to use for model responses')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Whether to overwrite existing results')
@click.option('--skip-model-eval/--no-skip-model-eval', default=False,
              help='Whether to skip model evaluation and only process existing results')
def evaluate(model_name: str, method: str, mode: str, data_name: str,
            labeled_path: str, raw_text_name: str, gpu_device: int,
            use_vague: bool, parser_type: str, overwrite: bool,
            skip_model_eval: bool):
    """Evaluate the model on test data using specified pipeline configuration."""
    logger.info(f"Starting evaluation with model {model_name} using {method} method in {mode} mode")
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    
    # Configure prompt parameters based on mode
    prompt_params = ['text', 'relations']
    suffix_path = 'completion'
    
    # Set up parser
    parser = JsonParser if parser_type == 'json' else LabelParser
    
    # Configure prompt based on method and parser type
    is_few_shot = (method == 'few-shot')
    if issubclass(parser, LabelParser):
        prompt = PairwisePrompt(use_few_shot=is_few_shot, use_vague=use_vague)
    else:
        prompt = MultiEvents(use_few_shot=is_few_shot, use_vague=use_vague, provide_justification=False)
    
    # Initialize model
    try:
        if 'gpt' in model_name.lower():
            model = OpenAIClient(model_name=model_name, use_formate=False, parser=parser, use_dot_graph=(mode=='multi'))
        else:
            model = HuggingfaceClient(model_name=model_name, device=gpu_device, parser=parser)
        
        # Run evaluation pipeline
        results_path = main(
            model_name=model_name,
            method=method,
            model=model,
            prompt_params=prompt_params,
            raw_text_name=raw_text_name,
            data_name=data_name,
            prompt=prompt,
            suffix_path=suffix_path,
            mode=mode,
            overwrite=overwrite,
            use_vague=use_vague,
            skip_model_eval=skip_model_eval
        )
        
        # Get summary results
        results_df = get_summary_results(
            model_name=model_name,
            method=method,
            labeled_path=Path(labeled_path),
            results_path=results_path,
            suffix_path=suffix_path,
            use_vague=use_vague
        )
        
        # Save results
        results_metrics_path = Path('results') / 'final_metrics' / method / results_path.name
        results_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_metrics_path, index=False)
        
        logger.info(f"Evaluation completed. Results saved to {results_metrics_path}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

@cli.command()
@click.option('--train-data', type=click.Path(exists=True), required=True,
              help='Path to the training data')
@click.option('--val-data', type=click.Path(exists=True), required=True,
              help='Path to the validation data')
@click.option('--model-dir', type=click.Path(), required=True,
              help='Directory to save the trained model')
@click.option('--epochs', type=int, default=10,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=32,
              help='Training batch size')
def train(train_data: str, val_data: str, model_dir: str,
          epochs: int, batch_size: int):
    """Train the model on the provided data."""
    logger.info(f"Training model with {epochs} epochs and batch size {batch_size}")
    # TODO: Implement training logic

@cli.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='Input file containing bracketed cycles')
@click.option('--output-file', type=click.Path(), required=True,
              help='Output file for processed cycles')
def brack_cycles(input_file: str, output_file: str):
    """Process bracketed cycles from input file."""
    logger.info(f"Processing bracketed cycles from {input_file}")
    # TODO: Implement bracketed cycles processing logic

if __name__ == '__main__':
    cli()
