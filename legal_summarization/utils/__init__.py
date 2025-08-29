"""Utilities module for legal summarization."""

"""Utilities module for legal summarization."""

from .json_storage import JSONStorageManager
from .preprocessing import LegalDocumentPreprocessor, preprocess_batch, extract_key_phrases

# Import evaluation only if dependencies are available
try:
    from .evaluation import LegalSummarizationEvaluator, EvaluationMetrics, calculate_summary_statistics
    __all__ = [
        "JSONStorageManager",
        "LegalSummarizationEvaluator", 
        "EvaluationMetrics",
        "calculate_summary_statistics",
        "LegalDocumentPreprocessor",
        "preprocess_batch",
        "extract_key_phrases"
    ]
except ImportError:
    __all__ = [
        "JSONStorageManager",
        "LegalDocumentPreprocessor",
        "preprocess_batch", 
        "extract_key_phrases"
    ]