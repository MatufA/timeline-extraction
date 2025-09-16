"""
Graph links counting module for analyzing temporal relation extraction results.

This module provides functions to count and analyze graph links from temporal
relation extraction results across different models and methods.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def count_graph_links_by_docid(
    results_path: Union[str, Path],
    model_name: str,
    method_name: str
) -> pd.DataFrame:
    """
    Count unique graph links per document ID from a results CSV file.
    
    Args:
        results_path: Path to the results CSV file
        model_name: Name of the model used
        method_name: Name of the method used (e.g., 'zero-shot', 'few-shot')
        
    Returns:
        DataFrame with docid as index and count of unique links as values
    """
    results_df = pd.read_csv(results_path)
    result = (
        results_df.groupby("docid")
        .unique_id.nunique()
        .to_frame(name=f"{model_name}-{method_name}")
    )
    return result


def process_multiple_results(
    data_path: Union[str, Path],
    models: List[str] = None,
    methods: List[str] = None,
    results_subdir: str = "results"
) -> pd.DataFrame:
    """
    Process multiple result files and combine them into a single DataFrame.
    
    Args:
        data_path: Base path to the data directory
        models: List of model names to process (default: ['gpt-4o-mini', 'gpt-4o'])
        methods: List of methods to process (default: ['zero-shot', 'few-shot'])
        results_subdir: Subdirectory name containing results (default: 'results')
        
    Returns:
        Combined DataFrame with counts for each model-method combination
    """
    if models is None:
        models = ["gpt-4o-mini", "gpt-4o"]
    if methods is None:
        methods = ["zero-shot", "few-shot"]
    
    data_path = Path(data_path)
    trc_raw_path = data_path / "TRC"
    
    relations_dfs = []
    
    for model in models:
        for method in methods:
            results_path = (
                trc_raw_path
                / results_subdir
                / method
                / f"multi-te3-platinum-results-{model}-{method}-completion.csv"
            )
            
            if results_path.exists():
                result_df = count_graph_links_by_docid(results_path, model, method)
                relations_dfs.append(result_df)
            else:
                print(f"Warning: Results file not found: {results_path}")
    
    if relations_dfs:
        combined_result = pd.concat(relations_dfs, axis=1)
        return combined_result
    else:
        print("No valid result files found")
        return pd.DataFrame()


def analyze_graph_links_summary(
    combined_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Generate summary statistics for the combined graph links data.
    
    Args:
        combined_df: Combined DataFrame from process_multiple_results
        output_path: Optional path to save summary statistics
        
    Returns:
        DataFrame with summary statistics
    """
    summary_stats = combined_df.describe()
    
    # Add additional statistics
    summary_stats.loc['total_docs'] = combined_df.shape[0]
    summary_stats.loc['missing_values'] = combined_df.isnull().sum()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_stats.to_csv(output_path)
        print(f"Summary statistics saved to: {output_path}")
    
    return summary_stats


def get_default_data_path() -> Path:
    """
    Get the default data path relative to the current file location.
    
    Returns:
        Path object pointing to the default data directory
    """
    return Path(__file__).parent.parent.parent / "data"


def main():
    """
    Example usage of the graph links counting functions.
    This demonstrates how to process multiple result files and analyze them.
    """
    # Use default data path or specify custom path
    data_path = get_default_data_path()
    
    print(f"Processing results from: {data_path}")
    
    # Process all result files
    combined_results = process_multiple_results(data_path)
    
    if not combined_results.empty:
        print("\nCombined Results:")
        print(combined_results.head())
        
        # Generate summary statistics
        summary = analyze_graph_links_summary(
            combined_results,
            data_path / "graph_links_summary.csv"
        )
        
        print("\nSummary Statistics:")
        print(summary)
        
        # Display basic info
        print(f"\nTotal documents processed: {combined_results.shape[0]}")
        print(f"Model-method combinations: {combined_results.shape[1]}")
        
    else:
        print("No results to process. Please check your data path and file structure.")


if __name__ == "__main__":
    main()
