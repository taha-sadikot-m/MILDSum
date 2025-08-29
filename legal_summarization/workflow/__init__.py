"""Workflow module for legal summarization."""

from .states import (
    LegalSummarizationState,
    CaseMetadata,
    MajorEvent,
    SubEvent,
    LegalReasoning,
    SummaryComponents,
    DocumentSections,
    QualityMetrics,
    EvaluationScores
)

__all__ = [
    "LegalSummarizationState",
    "CaseMetadata", 
    "MajorEvent",
    "SubEvent",
    "LegalReasoning",
    "SummaryComponents",
    "DocumentSections",
    "QualityMetrics",
    "EvaluationScores"
]

# Import workflow only if dependencies are available
try:
    from .graph import LegalSummarizationWorkflow
    __all__.append("LegalSummarizationWorkflow")
except ImportError:
    pass  # LangGraph not available