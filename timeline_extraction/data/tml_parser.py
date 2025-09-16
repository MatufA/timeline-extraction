"""
TimeML (TML) parser module for extracting temporal relations from TimeML files.

This module provides functions to parse TimeML files and extract temporal relations
into structured data formats suitable for machine learning tasks.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from bs4 import BeautifulSoup


def replace_eid(text: str, exclude_ids: List[str]) -> str:
    """
    Replace event IDs in text with formatted markers for excluded IDs.
    
    Args:
        text: Input text containing event IDs in format "ei{id}:{word}"
        exclude_ids: List of event IDs to mark with special formatting
        
    Returns:
        Text with event IDs replaced according to exclusion rules
    """
    pattern = r"ei(\d+):(\w+)"

    def replace_func(match):
        ei_id = match.group(1)
        word = match.group(2)

        if f"ei{ei_id}" in exclude_ids:
            return f"[e{ei_id}]{word}[/e{ei_id}]"
        else:
            return word

    # Use re.sub with a replacement function
    result = re.sub(pattern, replace_func, text)
    return result


def parse_tml_to_dataframe(tml_file_path: Union[str, Path], docid: str) -> Tuple[pd.DataFrame, int, int, List[Dict]]:
    """
    Parse a TimeML (TML) file into a pandas DataFrame with temporal relations.

    Args:
        tml_file_path: Path to the TML file
        docid: Document identifier

    Returns:
        Tuple containing:
        - DataFrame with columns ['docid', 'text', 'eiid1', 'eiid2', 'relation', 'unique_id']
        - Number of events found
        - Number of temporal links found
        - List of records in JSONL format for evaluation
    """
    jsonl_format = []
    
    # Parse the TML file
    with Path(tml_file_path).open("r") as f:
        data = f.read()
    doc = BeautifulSoup(data, features="lxml")

    # Create mapping from event IDs to event instance IDs
    mapping = {}
    for d in doc.find_all("makeinstance"):
        if d.get("eventid") not in mapping:
            mapping[d.get("eventid")] = d.get("eiid")

    # Extract and clean text content
    text = doc.find(name="text")
    try:
        [
            d.replace_with(
                f'{mapping[d.get("eid")] if d.get("eid") in mapping else d.get("eid")}:{d.get_text()} '
            )
            for d in text.find_all("event")
        ]
        [d.replace_with(f"{d.get_text()} ") for d in text.find_all("timex3")]
        text = text.get_text().strip()
    except AttributeError:
        print(f"Error processing document {docid}")
        print(f"Text content: {text}")
        raise

    events = set()
    unique_ids = set()

    # Extract TLINKs (temporal relations)
    tlinks = []
    for tlink in doc.find_all("tlink"):
        eiid1 = tlink.get("eventinstanceid", None)
        eiid2 = tlink.get("relatedtoeventinstance", None)
        relation = tlink.get("reltype", None)

        # Append only if all required attributes are present
        if eiid1 and eiid2 and relation:
            events.add(eiid1)
            events.add(eiid2)
            unique_id = "-".join(sorted([eiid1, eiid2]))
            tlinks.append((docid, text, eiid1, eiid2, relation, unique_id))

            if unique_id not in unique_ids:
                jsonl_format.append(
                    {
                        "text": text,
                        "relations": [(eiid1, eiid2)],
                        "docid": docid,
                        "true_label": relation.lower(),
                    }
                )
                unique_ids.add(unique_id)

    # Create a DataFrame from the extracted TLINKs
    df = pd.DataFrame(
        tlinks, columns=["docid", "text", "eiid1", "eiid2", "relation", "unique_id"]
    )

    return df, len(events), len(tlinks), jsonl_format


def process_timebank_directory(
    timebank_path: Union[str, Path],
    output_dir: Union[str, Path] = None
) -> Tuple[pd.DataFrame, List[Dict], Dict[str, int]]:
    """
    Process all TML files in a TimeBank directory and extract temporal relations.
    
    Args:
        timebank_path: Path to directory containing TML files
        output_dir: Optional directory to save output files
        
    Returns:
        Tuple containing:
        - Combined DataFrame of all temporal relations
        - List of all records for evaluation
        - Dictionary with statistics (total_events, total_tlinks, num_files)
    """
    timebank_path = Path(timebank_path)
    datasets = []
    total_events = []
    total_tlinks = []
    records_to_evaluate = []
    
    for file in timebank_path.glob("*.tml"):
        df, n_events, n_tlinks, jsonl_format = parse_tml_to_dataframe(
            file, file.name.replace(".tml", "")
        )
        datasets.append(df)
        total_events.append(n_events)
        total_tlinks.append(n_tlinks)
        records_to_evaluate.extend(jsonl_format)
    
    # Combine all datasets
    df_all = pd.concat(datasets, ignore_index=True)
    
    # Save combined dataset if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_dir / "timebank_dataset.csv", index=False)
    
    stats = {
        "total_events": sum(total_events),
        "total_tlinks": sum(total_tlinks),
        "num_files": len(datasets)
    }
    
    return df_all, records_to_evaluate, stats


def split_train_test_data(
    records: List[Dict],
    test_files: List[str],
    train_output_path: Union[str, Path] = None,
    test_output_path: Union[str, Path] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records into training and test sets based on document IDs.
    
    Args:
        records: List of records to split
        test_files: List of document IDs to use for testing
        train_output_path: Optional path to save training data
        test_output_path: Optional path to save test data
        
    Returns:
        Tuple of (train_records, test_records)
    """
    train_records = []
    test_records = []
    
    for record in records:
        if record["docid"] in test_files:
            test_records.append(record)
        else:
            train_records.append(record)
    
    # Save to files if paths are provided
    if train_output_path:
        Path(train_output_path).write_text(json.dumps(train_records, indent=2))
    
    if test_output_path:
        Path(test_output_path).write_text(json.dumps(test_records, indent=2))
    
    return train_records, test_records


def get_default_test_files() -> List[str]:
    """
    Get the default list of test files used in the original TimeBank evaluation.
    
    Returns:
        List of document IDs for test set
    """
    return [
        "APW19980227.0489",
        "APW19980227.0494",
        "APW19980308.0201",
        "APW19980418.0210",
        "CNN19980126.1600.1104",
        "CNN19980213.2130.0155",
        "NYT19980402.0453",
        "PRI19980115.2000.0186",
        "PRI19980306.2000.1675",
    ]


def main():
    """
    Example usage of the TML parser functions.
    This demonstrates how to process TimeBank data and create train/test splits.
    """
    # Example paths - update these for your specific use case
    timebank_path = Path("/home/adiel/nt/corpus/timebank/nt_converted_to_tml/a1")
    output_dir = Path("/home/adiel/full-temporal-relation/data")
    
    # Process all TML files in the directory
    df_all, records_to_evaluate, stats = process_timebank_directory(
        timebank_path, output_dir
    )
    
    print(f"Processed {stats['num_files']} files")
    print(f"Total events: {stats['total_events']}")
    print(f"Total temporal links: {stats['total_tlinks']}")
    
    # Split into train/test sets
    test_files = get_default_test_files()
    train_records, test_records = split_train_test_data(
        records_to_evaluate,
        test_files,
        output_dir / "narrativetime_a1_train.json",
        output_dir / "narrativetime_a1_test.json"
    )
    
    print(f"Training records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")


if __name__ == "__main__":
    main()
