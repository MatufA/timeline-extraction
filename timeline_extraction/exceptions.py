"""Custom exceptions for timeline extraction."""


class TimelineExtractionError(Exception):
    """Base exception for timeline extraction errors."""

    pass


class ConfigurationError(TimelineExtractionError):
    """Exception raised for configuration-related errors."""

    pass


class DataError(TimelineExtractionError):
    """Exception raised for data-related errors."""

    pass


class ModelError(TimelineExtractionError):
    """Exception raised for model-related errors."""

    pass


class EvaluationError(TimelineExtractionError):
    """Exception raised for evaluation-related errors."""

    pass


class ValidationError(TimelineExtractionError):
    """Exception raised for validation errors."""

    pass


class FileError(TimelineExtractionError):
    """Exception raised for file-related errors."""

    pass


class APIError(TimelineExtractionError):
    """Exception raised for API-related errors."""

    pass


class ParsingError(TimelineExtractionError):
    """Exception raised for parsing errors."""

    pass


class CycleDetectionError(TimelineExtractionError):
    """Exception raised for cycle detection errors."""

    pass
