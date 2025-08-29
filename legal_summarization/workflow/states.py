"""State management for Legal Summarization workflow."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel
import json


class CaseMetadata(BaseModel):
    """Metadata for a legal case."""
    case_number: str = ""
    court: str = ""
    date: str = ""
    parties: List[str] = field(default_factory=list)
    judges: List[str] = field(default_factory=list)
    case_type: str = ""


class SubEvent(BaseModel):
    """Sub-event within a major legal event."""
    sub_event_id: str
    description: str
    evidence: List[str] = field(default_factory=list)
    legal_references: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    participants: List[str] = field(default_factory=list)


class MajorEvent(BaseModel):
    """Major legal event in the case."""
    event_id: str
    title: str
    description: str
    timestamp: Optional[str] = None
    legal_significance: str
    event_type: str  # procedural, substantive, evidential, etc.
    sub_events: List[SubEvent] = field(default_factory=list)


class LegalReasoning(BaseModel):
    """Legal reasoning components."""
    key_issues: List[str] = field(default_factory=list)
    precedents_cited: List[str] = field(default_factory=list)
    legal_principles: List[str] = field(default_factory=list)
    reasoning_flow: List[str] = field(default_factory=list)


class SummaryComponents(BaseModel):
    """Components of the legal summary."""
    facts: str = ""
    issues: str = ""
    reasoning: str = ""
    conclusion: str = ""


class DocumentSections(BaseModel):
    """Sections identified in the legal document."""
    facts: str = ""
    legal_issues: str = ""
    reasoning: str = ""
    conclusion: str = ""
    procedural_history: str = ""


class QualityMetrics(BaseModel):
    """Quality evaluation metrics."""
    legal_accuracy: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    clarity: float = 0.0
    legal_language: float = 0.0
    overall_score: float = 0.0
    improvement_suggestions: List[str] = field(default_factory=list)


class EvaluationScores(BaseModel):
    """Evaluation scores for summary quality."""
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    legal_metrics: QualityMetrics = field(default_factory=QualityMetrics)


@dataclass
class LegalSummarizationState:
    """
    Complete state for the legal summarization workflow.
    
    This state is passed between agents in the LangGraph workflow,
    with each agent updating relevant sections.
    """
    
    # Input document
    document_text: str = ""
    document_id: str = ""
    
    # Preprocessing results
    case_metadata: CaseMetadata = field(default_factory=CaseMetadata)
    document_sections: DocumentSections = field(default_factory=DocumentSections)
    preprocessing_notes: List[str] = field(default_factory=list)
    
    # Event extraction results
    major_events: List[MajorEvent] = field(default_factory=list)
    event_timeline: List[str] = field(default_factory=list)
    
    # Knowledge graph components
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    legal_reasoning: LegalReasoning = field(default_factory=LegalReasoning)
    
    # Summary generation
    summary_components: SummaryComponents = field(default_factory=SummaryComponents)
    generated_summary: str = ""
    target_length: int = 500
    
    # Quality evaluation
    evaluation_scores: EvaluationScores = field(default_factory=EvaluationScores)
    quality_feedback: List[str] = field(default_factory=list)
    
    # Workflow metadata
    current_agent: str = ""
    processing_stage: str = "initial"
    error_messages: List[str] = field(default_factory=list)
    retry_count: int = 0
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "document_text": self.document_text,
            "document_id": self.document_id,
            "case_metadata": self.case_metadata.model_dump() if self.case_metadata else {},
            "document_sections": self.document_sections.model_dump() if self.document_sections else {},
            "preprocessing_notes": self.preprocessing_notes,
            "major_events": [event.model_dump() for event in self.major_events],
            "event_timeline": self.event_timeline,
            "knowledge_graph": self.knowledge_graph,
            "legal_reasoning": self.legal_reasoning.model_dump() if self.legal_reasoning else {},
            "summary_components": self.summary_components.model_dump() if self.summary_components else {},
            "generated_summary": self.generated_summary,
            "target_length": self.target_length,
            "evaluation_scores": self.evaluation_scores.model_dump() if self.evaluation_scores else {},
            "quality_feedback": self.quality_feedback,
            "current_agent": self.current_agent,
            "processing_stage": self.processing_stage,
            "error_messages": self.error_messages,
            "retry_count": self.retry_count,
            "workflow_metadata": self.workflow_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalSummarizationState":
        """Create state from dictionary."""
        state = cls()
        
        # Basic fields
        state.document_text = data.get("document_text", "")
        state.document_id = data.get("document_id", "")
        state.preprocessing_notes = data.get("preprocessing_notes", [])
        state.event_timeline = data.get("event_timeline", [])
        state.knowledge_graph = data.get("knowledge_graph", {})
        state.generated_summary = data.get("generated_summary", "")
        state.target_length = data.get("target_length", 500)
        state.quality_feedback = data.get("quality_feedback", [])
        state.current_agent = data.get("current_agent", "")
        state.processing_stage = data.get("processing_stage", "initial")
        state.error_messages = data.get("error_messages", [])
        state.retry_count = data.get("retry_count", 0)
        state.workflow_metadata = data.get("workflow_metadata", {})
        
        # Pydantic models
        if data.get("case_metadata"):
            state.case_metadata = CaseMetadata(**data["case_metadata"])
        
        if data.get("document_sections"):
            state.document_sections = DocumentSections(**data["document_sections"])
        
        if data.get("major_events"):
            state.major_events = [MajorEvent(**event) for event in data["major_events"]]
        
        if data.get("legal_reasoning"):
            state.legal_reasoning = LegalReasoning(**data["legal_reasoning"])
        
        if data.get("summary_components"):
            state.summary_components = SummaryComponents(**data["summary_components"])
        
        if data.get("evaluation_scores"):
            state.evaluation_scores = EvaluationScores(**data["evaluation_scores"])
        
        return state
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the state."""
        self.error_messages.append(error_message)
    
    def increment_retry(self) -> None:
        """Increment the retry counter."""
        self.retry_count += 1
    
    def update_stage(self, stage: str, agent: str = "") -> None:
        """Update the processing stage and current agent."""
        self.processing_stage = stage
        if agent:
            self.current_agent = agent
    
    def get_json_output(self) -> str:
        """Get JSON representation of the complete workflow output."""
        output = {
            "case_metadata": self.case_metadata.model_dump(),
            "major_events": [event.model_dump() for event in self.major_events],
            "legal_reasoning": self.legal_reasoning.model_dump(),
            "summary_components": self.summary_components.model_dump(),
            "generated_summary": self.generated_summary,
            "evaluation_scores": self.evaluation_scores.model_dump(),
            "knowledge_graph": self.knowledge_graph,
        }
        return json.dumps(output, indent=2, ensure_ascii=False)