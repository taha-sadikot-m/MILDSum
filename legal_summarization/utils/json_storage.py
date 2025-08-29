"""JSON storage utilities for legal summarization workflow."""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from ..workflow.states import LegalSummarizationState
from ..config.settings import settings

logger = logging.getLogger(__name__)


class JSONStorageManager:
    """Manager for storing and retrieving workflow results in JSON format."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the JSON storage manager."""
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.results_dir = self.output_dir / "results"
        self.events_dir = self.output_dir / "events"
        self.summaries_dir = self.output_dir / "summaries"
        self.evaluations_dir = self.output_dir / "evaluations"
        
        for dir_path in [self.results_dir, self.events_dir, self.summaries_dir, self.evaluations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_workflow_result(
        self, 
        state: LegalSummarizationState, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save complete workflow result to JSON file.
        
        Args:
            state: Final workflow state
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"workflow_result_{state.document_id}_{timestamp}.json"
            
            file_path = self.results_dir / filename
            
            # Prepare complete result data
            result_data = {
                "metadata": {
                    "document_id": state.document_id,
                    "processing_timestamp": datetime.now().isoformat(),
                    "workflow_version": "1.0.0",
                    "target_length": state.target_length,
                    "processing_stage": state.processing_stage,
                    "retry_count": state.retry_count
                },
                "case_metadata": state.case_metadata.model_dump() if state.case_metadata else {},
                "document_sections": state.document_sections.model_dump() if state.document_sections else {},
                "major_events": [event.model_dump() for event in state.major_events],
                "legal_reasoning": state.legal_reasoning.model_dump() if state.legal_reasoning else {},
                "knowledge_graph": state.knowledge_graph,
                "summary_components": state.summary_components.model_dump() if state.summary_components else {},
                "generated_summary": state.generated_summary,
                "evaluation_scores": state.evaluation_scores.model_dump() if state.evaluation_scores else {},
                "quality_feedback": state.quality_feedback,
                "error_messages": state.error_messages,
                "workflow_metadata": state.workflow_metadata
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Workflow result saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save workflow result: {str(e)}")
            raise
    
    def save_events_json(
        self, 
        state: LegalSummarizationState, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save extracted events in structured JSON format.
        
        Args:
            state: Workflow state containing events
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"events_{state.document_id}_{timestamp}.json"
            
            file_path = self.events_dir / filename
            
            # Prepare events data according to specified schema
            events_data = {
                "case_metadata": {
                    "case_number": state.case_metadata.case_number if state.case_metadata else "",
                    "court": state.case_metadata.court if state.case_metadata else "",
                    "date": state.case_metadata.date if state.case_metadata else "",
                    "parties": state.case_metadata.parties if state.case_metadata else []
                },
                "major_events": [],
                "legal_reasoning": {
                    "key_issues": state.legal_reasoning.key_issues if state.legal_reasoning else [],
                    "precedents_cited": state.legal_reasoning.precedents_cited if state.legal_reasoning else [],
                    "legal_principles": state.legal_reasoning.legal_principles if state.legal_reasoning else []
                },
                "summary_components": {
                    "facts": state.summary_components.facts if state.summary_components else "",
                    "issues": state.summary_components.issues if state.summary_components else "",
                    "reasoning": state.summary_components.reasoning if state.summary_components else "",
                    "conclusion": state.summary_components.conclusion if state.summary_components else ""
                }
            }
            
            # Add major events with sub-events
            for event in state.major_events:
                event_data = {
                    "event_id": event.event_id,
                    "title": event.title,
                    "description": event.description,
                    "timestamp": event.timestamp or "",
                    "legal_significance": event.legal_significance,
                    "sub_events": []
                }
                
                # Add sub-events
                for sub_event in event.sub_events:
                    sub_event_data = {
                        "sub_event_id": sub_event.sub_event_id,
                        "description": sub_event.description,
                        "evidence": sub_event.evidence,
                        "legal_references": sub_event.legal_references
                    }
                    event_data["sub_events"].append(sub_event_data)
                
                events_data["major_events"].append(event_data)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Events JSON saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save events JSON: {str(e)}")
            raise
    
    def save_summary_json(
        self, 
        state: LegalSummarizationState, 
        filename: Optional[str] = None
    ) -> str:
        """
        Save generated summary with metadata.
        
        Args:
            state: Workflow state containing summary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"summary_{state.document_id}_{timestamp}.json"
            
            file_path = self.summaries_dir / filename
            
            # Prepare summary data
            summary_data = {
                "document_id": state.document_id,
                "generation_timestamp": datetime.now().isoformat(),
                "case_info": {
                    "case_number": state.case_metadata.case_number if state.case_metadata else "",
                    "court": state.case_metadata.court if state.case_metadata else "",
                    "parties": state.case_metadata.parties if state.case_metadata else []
                },
                "summary": {
                    "text": state.generated_summary,
                    "word_count": len(state.generated_summary.split()) if state.generated_summary else 0,
                    "target_length": state.target_length
                },
                "summary_components": {
                    "facts": state.summary_components.facts if state.summary_components else "",
                    "issues": state.summary_components.issues if state.summary_components else "",
                    "reasoning": state.summary_components.reasoning if state.summary_components else "",
                    "conclusion": state.summary_components.conclusion if state.summary_components else ""
                },
                "evaluation": {
                    "rouge_scores": {
                        "rouge_1": state.evaluation_scores.rouge_1 if state.evaluation_scores else 0.0,
                        "rouge_2": state.evaluation_scores.rouge_2 if state.evaluation_scores else 0.0,
                        "rouge_l": state.evaluation_scores.rouge_l if state.evaluation_scores else 0.0
                    },
                    "bert_score": state.evaluation_scores.bert_score if state.evaluation_scores else 0.0,
                    "legal_quality": {
                        "overall_score": state.evaluation_scores.legal_metrics.overall_score if state.evaluation_scores and state.evaluation_scores.legal_metrics else 0.0,
                        "legal_accuracy": state.evaluation_scores.legal_metrics.legal_accuracy if state.evaluation_scores and state.evaluation_scores.legal_metrics else 0.0,
                        "completeness": state.evaluation_scores.legal_metrics.completeness if state.evaluation_scores and state.evaluation_scores.legal_metrics else 0.0
                    }
                },
                "quality_feedback": state.quality_feedback
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary JSON saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save summary JSON: {str(e)}")
            raise
    
    def load_workflow_result(self, file_path: str) -> Dict[str, Any]:
        """
        Load workflow result from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Workflow result data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load workflow result: {str(e)}")
            raise
    
    def list_results(self, pattern: str = "*.json") -> List[str]:
        """
        List available result files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of file paths
        """
        try:
            return [str(f) for f in self.results_dir.glob(pattern)]
        except Exception as e:
            logger.error(f"Failed to list results: {str(e)}")
            return []
    
    def get_latest_result(self, document_id: Optional[str] = None) -> Optional[str]:
        """
        Get the latest result file for a document.
        
        Args:
            document_id: Optional document ID to filter by
            
        Returns:
            Path to latest result file
        """
        try:
            pattern = f"*{document_id}*.json" if document_id else "*.json"
            files = list(self.results_dir.glob(pattern))
            
            if not files:
                return None
            
            # Sort by modification time and return latest
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
            
        except Exception as e:
            logger.error(f"Failed to get latest result: {str(e)}")
            return None
    
    def export_batch_results(
        self, 
        results: List[LegalSummarizationState], 
        filename: Optional[str] = None
    ) -> str:
        """
        Export multiple workflow results to a single JSON file.
        
        Args:
            results: List of workflow states
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_results_{timestamp}.json"
            
            file_path = self.output_dir / filename
            
            batch_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_documents": len(results),
                "results": []
            }
            
            for state in results:
                result_summary = {
                    "document_id": state.document_id,
                    "success": bool(state.generated_summary and not state.error_messages),
                    "summary_length": len(state.generated_summary.split()) if state.generated_summary else 0,
                    "events_count": len(state.major_events),
                    "overall_quality": state.evaluation_scores.legal_metrics.overall_score if state.evaluation_scores and state.evaluation_scores.legal_metrics else 0.0,
                    "errors": state.error_messages
                }
                batch_data["results"].append(result_summary)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch results exported to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export batch results: {str(e)}")
            raise