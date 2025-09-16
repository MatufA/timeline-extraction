"""Utility functions for timeline extraction."""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from timeline_extraction.exceptions import FileError


def setup_logging(
    level: str = "INFO", log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON text with fallback options.

    Args:
        text: JSON text to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            # Remove extra commas and fix common issues
            fixed_text = re.sub(r",\s*}", "}", text)
            fixed_text = re.sub(r",\s*]", "]", fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON: {text[:100]}...")
            return default


def extract_event_ids(text: str, pattern: str = r"ei(\d+)") -> List[str]:
    """Extract event IDs from text using regex pattern.

    Args:
        text: Text to search for event IDs
        pattern: Regex pattern for event IDs

    Returns:
        List of found event IDs
    """
    matches = re.findall(pattern, text)
    return [f"ei{match}" for match in matches]


def normalize_event_id(event_id: str) -> str:
    """Normalize event ID format.

    Args:
        event_id: Event ID to normalize

    Returns:
        Normalized event ID
    """
    # Remove any non-alphanumeric characters except 'ei' prefix
    normalized = re.sub(r"[^a-zA-Z0-9]", "", event_id)

    # Ensure it starts with 'ei' if it's a number
    if normalized.isdigit():
        normalized = f"ei{normalized}"
    elif not normalized.startswith("ei") and normalized[2:].isdigit():
        normalized = f"ei{normalized[2:]}"

    return normalized


def create_unique_id(eiid1: str, eiid2: str) -> str:
    """Create a unique identifier for a pair of events.

    Args:
        eiid1: First event ID
        eiid2: Second event ID

    Returns:
        Unique identifier string
    """
    return "-".join(sorted([eiid1, eiid2]))


def validate_temporal_relation(relation: str) -> bool:
    """Validate if a relation is a valid temporal relation.

    Args:
        relation: Relation string to validate

    Returns:
        True if valid, False otherwise
    """
    valid_relations = {"BEFORE", "AFTER", "EQUAL", "VAGUE"}
    return relation.upper() in valid_relations


def normalize_relation(relation: str) -> str:
    """Normalize temporal relation to standard format.

    Args:
        relation: Relation to normalize

    Returns:
        Normalized relation string
    """
    relation = relation.upper().strip()

    # Handle common variations
    if relation in {"BEFORE", "B", "PREVIOUS", "EARLIER"}:
        return "BEFORE"
    elif relation in {"AFTER", "A", "NEXT", "LATER"}:
        return "AFTER"
    elif relation in {"EQUAL", "E", "SAME", "SIMULTANEOUS"}:
        return "EQUAL"
    elif relation in {"VAGUE", "V", "UNKNOWN", "UNCLEAR"}:
        return "VAGUE"
    else:
        return relation


def calculate_metrics(
    y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of unique labels

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if labels is None:
        labels = list(set(y_true + y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate micro averages
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": {
            label: {"precision": p, "recall": r, "f1": f, "support": s}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
    }


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with error handling.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation

    Raises:
        FileError: If file cannot be saved
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        raise FileError(f"Failed to save JSON to {file_path}: {e}") from e


def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from JSON file with error handling.

    Args:
        file_path: Input file path

    Returns:
        Loaded data

    Raises:
        FileError: If file cannot be loaded
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileError(f"File does not exist: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        raise FileError(f"Failed to load JSON from {file_path}: {e}") from e


def save_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to CSV file with error handling.

    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for to_csv

    Raises:
        FileError: If file cannot be saved
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    try:
        df.to_csv(file_path, index=False, **kwargs)
    except Exception as e:
        logging.error(f"Failed to save CSV to {file_path}: {e}")
        raise FileError(f"Failed to save CSV to {file_path}: {e}") from e


def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load DataFrame from CSV file with error handling.

    Args:
        file_path: Input file path
        **kwargs: Additional arguments for read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileError: If file cannot be loaded
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileError(f"File does not exist: {file_path}")

    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logging.error(f"Failed to load CSV from {file_path}: {e}")
        raise FileError(f"Failed to load CSV from {file_path}: {e}") from e


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list.

    Args:
        nested_list: Nested list to flatten

    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension from path.

    Args:
        file_path: File path

    Returns:
        File extension (without dot)
    """
    return Path(file_path).suffix.lstrip(".")


def is_valid_file(
    file_path: Union[str, Path], extensions: Optional[List[str]] = None
) -> bool:
    """Check if file exists and has valid extension.

    Args:
        file_path: File path to check
        extensions: List of valid extensions (optional)

    Returns:
        True if file is valid, False otherwise
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False

    if extensions is not None:
        return get_file_extension(file_path) in extensions

    return True


def retry_on_exception(
    max_retries: int = 3, delay: float = 1.0, exceptions: Tuple = (Exception,)
):
    """Decorator to retry function on exception.

    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logging.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
            return None

        return wrapper

    return decorator
