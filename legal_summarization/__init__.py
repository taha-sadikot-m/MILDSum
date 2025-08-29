"""
Legal Summarization Agentic Workflow
====================================

A LangGraph-based multi-agent system for generating abstractive summaries 
of Indian court case judgments using Google's Gemini LLM.

This module provides a comprehensive workflow for:
- Document preprocessing and structure analysis
- Event extraction and hierarchical organization
- Knowledge graph construction
- High-quality legal summary generation
- Quality evaluation and optimization
"""

__version__ = "1.0.0"
__author__ = "Legal AI Team"

# Import only core components for basic functionality
# Full workflow requires external dependencies (langgraph, google-generativeai)

try:
    from .workflow.graph import LegalSummarizationWorkflow
    _WORKFLOW_AVAILABLE = True
except ImportError:
    _WORKFLOW_AVAILABLE = False

try:
    from .models.gemini_client import GeminiClient
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# Always available core components
from .workflow.states import LegalSummarizationState, CaseMetadata, MajorEvent

__all__ = ["LegalSummarizationState", "CaseMetadata", "MajorEvent"]

if _WORKFLOW_AVAILABLE:
    __all__.append("LegalSummarizationWorkflow")

if _GEMINI_AVAILABLE:
    __all__.append("GeminiClient")