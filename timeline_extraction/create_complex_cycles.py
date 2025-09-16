"""
Complex cycles creation module for temporal relation extraction.

This module provides functions to process and join temporal relation cycles,
creating complex cycle structures from baseline temporal relation data.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd


def join_cycles(cycles: List[List[Tuple[str, str, str]]]) -> pd.DataFrame:
    """
    Join overlapping cycles by merging cycles that share common edges.
    
    Args:
        cycles: List of cycles, where each cycle is a list of tuples (e1, e2, relation)
        
    Returns:
        DataFrame with joined cycles containing columns: ['eiid1', 'eiid2', 'relation', 'unique_id', 'cycle_id']
    """
    cycles_together = []
    edges_sets = []

    for cycle in cycles:
        # Extract event instance IDs from the full event IDs
        cycle_data = [
            (e1.split("-")[1], e2.split("-")[1], relation) for e1, e2, relation in cycle
        ]
        current_edges = set()
        for e1, e2, _ in cycle_data:
            current_edges.add(e1)
            current_edges.add(e2)
        
        # Check for overlapping edges with existing cycles
        if edges_sets:
            merged = False
            for idx, edges in enumerate(edges_sets):
                if edges.intersection(current_edges):
                    cycles_together[idx].extend(cycle_data)
                    edges_sets[idx].update(current_edges - edges)
                    merged = True
                    break
            
            if not merged:
                edges_sets.append(current_edges)
                cycles_together.append(cycle_data)
        else:
            edges_sets.append(current_edges)
            cycles_together.append(cycle_data)

    # Convert joined cycles to DataFrame
    cycle_dfs = []
    for idx, cycle in enumerate(cycles_together):
        cycle_df = pd.DataFrame(cycle, columns=["eiid1", "eiid2", "relation"])
        eiid1_eiid2 = list(zip(cycle_df["eiid1"], cycle_df["eiid2"]))
        cycle_df["unique_id"] = [
            "-".join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2
        ]
        cycle_df["cycle_id"] = idx
        cycle_dfs.append(cycle_df)
    
    return pd.concat(cycle_dfs, ignore_index=True)


def process_cycles_data(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process cycles data from JSON file and create joined cycles DataFrame.
    
    Args:
        input_path: Path to the input JSON file containing cycles data
        output_path: Optional path to save the processed DataFrame as CSV
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with processed cycles data
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load cycles data
    with input_path.open("r") as f:
        data = json.load(f)
    
    joints_cycles_dfs = []
    
    for docid, cycles in data.items():
        if verbose:
            print(f"{docid}: {len(cycles)} cycles")
        
        # Join cycles for this document
        joint_cycles = join_cycles(cycles)
        joint_cycles["docid"] = docid
        joints_cycles_dfs.append(joint_cycles)
        
        if verbose:
            print(f"{docid}: {len(joint_cycles)} joint cycles")
    
    # Combine all document cycles
    df_all = pd.concat(joints_cycles_dfs, ignore_index=True)
    
    # Remove duplicates based on document ID, cycle ID, and unique ID
    df_all = df_all.drop_duplicates(subset=["docid", "cycle_id", "unique_id"])
    
    # Save to CSV if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_path, index=False)
        if verbose:
            print(f"Processed data saved to: {output_path}")
    
    return df_all


def analyze_cycles_statistics(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Generate statistics about the processed cycles data.
    
    Args:
        df: DataFrame containing processed cycles data
        
    Returns:
        Dictionary with various statistics about the cycles
    """
    stats = {
        "total_relations": len(df),
        "unique_documents": df["docid"].nunique(),
        "total_cycles": df["cycle_id"].nunique(),
        "unique_relations": df["unique_id"].nunique(),
        "avg_relations_per_doc": df.groupby("docid").size().mean(),
        "avg_relations_per_cycle": df.groupby(["docid", "cycle_id"]).size().mean(),
    }
    
    # Add relation type distribution
    relation_counts = df["relation"].value_counts()
    stats["relation_distribution"] = relation_counts.to_dict()
    
    return stats


def get_default_paths() -> Tuple[Path, Path]:
    """
    Get default input and output paths for cycles processing.
    
    Returns:
        Tuple of (input_path, output_path)
    """
    base_path = Path("/home/adiel/full-temporal-relation/data")
    input_path = base_path / "all_cycles_matres_by_baseline.json"
    output_path = base_path / "grouped_cycles_matres_by_baseline.csv"
    
    return input_path, output_path


def main():
    """
    Example usage of the complex cycles creation functions.
    This demonstrates how to process cycles data and create joined cycles.
    """
    # Get default paths
    input_path, output_path = get_default_paths()
    
    print(f"Processing cycles from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Process the cycles data
        df_all = process_cycles_data(input_path, output_path, verbose=True)
        
        # Generate and display statistics
        stats = analyze_cycles_statistics(df_all)
        
        print("\n=== Cycles Processing Statistics ===")
        print(f"Total relations: {stats['total_relations']}")
        print(f"Unique documents: {stats['unique_documents']}")
        print(f"Total cycles: {stats['total_cycles']}")
        print(f"Unique relations: {stats['unique_relations']}")
        print(f"Average relations per document: {stats['avg_relations_per_doc']:.2f}")
        print(f"Average relations per cycle: {stats['avg_relations_per_cycle']:.2f}")
        
        print("\n=== Relation Type Distribution ===")
        for relation, count in stats['relation_distribution'].items():
            print(f"{relation}: {count}")
        
        print(f"\nProcessed data shape: {df_all.shape}")
        print(f"Columns: {list(df_all.columns)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the input file exists and the path is correct.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
