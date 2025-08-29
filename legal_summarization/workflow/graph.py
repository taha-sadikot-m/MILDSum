"""LangGraph workflow for legal document summarization."""

import logging
from typing import Dict, Any, Optional, Callable
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..agents import (
    DocumentPreprocessorAgent,
    EventExtractorAgent, 
    SubEventAnalyzerAgent,
    KnowledgeBuilderAgent,
    SummaryGeneratorAgent,
    QualityEvaluatorAgent
)
from ..models.gemini_client import GeminiClient
from ..workflow.states import LegalSummarizationState
from ..config.settings import settings

logger = logging.getLogger(__name__)


class LegalSummarizationWorkflow:
    """
    LangGraph-based workflow for legal document summarization.
    
    This workflow orchestrates multiple specialized agents to process 
    legal documents and generate high-quality abstractive summaries.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the legal summarization workflow."""
        self.gemini_client = gemini_client or GeminiClient()
        
        # Initialize agents
        self.preprocessor = DocumentPreprocessorAgent(self.gemini_client)
        self.event_extractor = EventExtractorAgent(self.gemini_client)
        self.sub_event_analyzer = SubEventAnalyzerAgent(self.gemini_client)
        self.knowledge_builder = KnowledgeBuilderAgent(self.gemini_client)
        self.summary_generator = SummaryGeneratorAgent(self.gemini_client)
        self.quality_evaluator = QualityEvaluatorAgent(self.gemini_client)
        
        # Build the workflow graph
        self.workflow = self._build_workflow_graph()
        
        logger.info("Legal summarization workflow initialized")
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(LegalSummarizationState)
        
        # Add nodes for each agent
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("extract_events", self._extract_events_node)
        workflow.add_node("analyze_sub_events", self._analyze_sub_events_node)
        workflow.add_node("build_knowledge_graph", self._build_knowledge_graph_node)
        workflow.add_node("generate_summary", self._generate_summary_node)
        workflow.add_node("evaluate_quality", self._evaluate_quality_node)
        workflow.add_node("retry_processing", self._retry_processing_node)
        
        # Define the workflow edges
        workflow.set_entry_point("preprocess")
        
        # Linear flow with conditional error handling
        workflow.add_edge("preprocess", "extract_events")
        workflow.add_edge("extract_events", "analyze_sub_events")
        workflow.add_edge("analyze_sub_events", "build_knowledge_graph")
        workflow.add_edge("build_knowledge_graph", "generate_summary")
        workflow.add_edge("generate_summary", "evaluate_quality")
        
        # Conditional edges for error handling and quality control
        workflow.add_conditional_edges(
            "evaluate_quality",
            self._should_retry,
            {
                "retry": "retry_processing",
                "complete": END
            }
        )
        
        workflow.add_edge("retry_processing", "generate_summary")
        
        return workflow
    
    async def _preprocess_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Document preprocessing node."""
        try:
            logger.info("Starting document preprocessing")
            return await self.preprocessor.process(state)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            state.add_error(f"Preprocessing failed: {str(e)}")
            return state
    
    async def _extract_events_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Event extraction node."""
        try:
            logger.info("Starting event extraction")
            return await self.event_extractor.process(state)
        except Exception as e:
            logger.error(f"Error in event extraction: {str(e)}")
            state.add_error(f"Event extraction failed: {str(e)}")
            return state
    
    async def _analyze_sub_events_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Sub-event analysis node."""
        try:
            logger.info("Starting sub-event analysis")
            return await self.sub_event_analyzer.process(state)
        except Exception as e:
            logger.error(f"Error in sub-event analysis: {str(e)}")
            state.add_error(f"Sub-event analysis failed: {str(e)}")
            return state
    
    async def _build_knowledge_graph_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Knowledge graph construction node."""
        try:
            logger.info("Starting knowledge graph construction")
            return await self.knowledge_builder.process(state)
        except Exception as e:
            logger.error(f"Error in knowledge graph construction: {str(e)}")
            state.add_error(f"Knowledge graph construction failed: {str(e)}")
            return state
    
    async def _generate_summary_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Summary generation node."""
        try:
            logger.info("Starting summary generation")
            return await self.summary_generator.process(state)
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            state.add_error(f"Summary generation failed: {str(e)}")
            return state
    
    async def _evaluate_quality_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Quality evaluation node."""
        try:
            logger.info("Starting quality evaluation")
            return await self.quality_evaluator.process(state)
        except Exception as e:
            logger.error(f"Error in quality evaluation: {str(e)}")
            state.add_error(f"Quality evaluation failed: {str(e)}")
            return state
    
    async def _retry_processing_node(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """Retry processing node for handling quality issues."""
        logger.info(f"Retrying processing (attempt {state.retry_count + 1})")
        state.increment_retry()
        
        # Clear previous summary for regeneration
        state.generated_summary = ""
        state.quality_feedback.append(f"Retry attempt {state.retry_count}")
        
        return state
    
    def _should_retry(self, state: LegalSummarizationState) -> str:
        """Determine if processing should be retried based on quality scores."""
        # Don't retry if we've exceeded max retries
        if state.retry_count >= settings.workflow.max_retries:
            logger.info("Max retries exceeded, completing workflow")
            return "complete"
        
        # Don't retry if there are critical errors
        if state.error_messages:
            critical_errors = [
                error for error in state.error_messages 
                if any(keyword in error.lower() for keyword in ['api', 'connection', 'authentication'])
            ]
            if critical_errors:
                logger.info("Critical errors detected, completing workflow")
                return "complete"
        
        # Retry if quality is below threshold
        if state.evaluation_scores and state.evaluation_scores.legal_metrics:
            overall_score = state.evaluation_scores.legal_metrics.overall_score
            if overall_score < 6.0:  # Threshold for retry
                logger.info(f"Quality score {overall_score} below threshold, retrying")
                return "retry"
        
        # Otherwise complete the workflow
        return "complete"
    
    async def process_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        target_length: int = 500,
        **kwargs
    ) -> LegalSummarizationState:
        """
        Process a legal document through the complete workflow.
        
        Args:
            document_text: The legal document text to process
            document_id: Optional identifier for the document
            target_length: Target length for the summary in words
            **kwargs: Additional configuration options
            
        Returns:
            Final workflow state with generated summary and evaluation
        """
        try:
            # Initialize state
            initial_state = LegalSummarizationState(
                document_text=document_text,
                document_id=document_id or "unknown",
                target_length=target_length
            )
            
            # Add any additional metadata from kwargs
            initial_state.workflow_metadata.update(kwargs)
            
            logger.info(f"Starting workflow for document: {initial_state.document_id}")
            
            # Compile and run the workflow
            app = self.workflow.compile(checkpointer=MemorySaver())
            
            # Execute the workflow
            final_state = None
            config = {"configurable": {"thread_id": document_id or "default"}}
            
            async for state in app.astream(initial_state, config=config):
                final_state = state
                
                # Log progress
                current_values = list(state.values())[0] if state else None
                if current_values:
                    logger.info(f"Workflow stage: {current_values.processing_stage}")
            
            if final_state:
                # Extract the final state from the workflow output
                workflow_result = list(final_state.values())[0]
                logger.info(f"Workflow completed for document: {workflow_result.document_id}")
                return workflow_result
            else:
                raise ValueError("Workflow did not produce a final state")
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            # Return state with error information
            error_state = LegalSummarizationState(
                document_text=document_text,
                document_id=document_id or "unknown"
            )
            error_state.add_error(f"Workflow execution failed: {str(e)}")
            return error_state
    
    def process_document_sync(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        target_length: int = 500,
        **kwargs
    ) -> LegalSummarizationState:
        """
        Synchronous wrapper for document processing.
        
        Args:
            document_text: The legal document text to process
            document_id: Optional identifier for the document  
            target_length: Target length for the summary in words
            **kwargs: Additional configuration options
            
        Returns:
            Final workflow state with generated summary and evaluation
        """
        try:
            # Run the async workflow
            return asyncio.run(self.process_document(
                document_text, document_id, target_length, **kwargs
            ))
        except Exception as e:
            logger.error(f"Synchronous workflow execution failed: {str(e)}")
            error_state = LegalSummarizationState(
                document_text=document_text,
                document_id=document_id or "unknown"
            )
            error_state.add_error(f"Synchronous workflow execution failed: {str(e)}")
            return error_state
    
    async def process_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 3,
        **kwargs
    ) -> List[LegalSummarizationState]:
        """
        Process multiple documents in batches.
        
        Args:
            documents: List of document dictionaries with 'text' and optional 'id'
            batch_size: Number of documents to process in parallel
            **kwargs: Additional configuration options
            
        Returns:
            List of workflow states for each processed document
        """
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Process batch in parallel
            tasks = [
                self.process_document(
                    doc.get("text", ""),
                    doc.get("id", f"doc_{i + j}"),
                    doc.get("target_length", 500),
                    **kwargs
                )
                for j, doc in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the batch
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {str(result)}")
                    error_state = LegalSummarizationState()
                    error_state.add_error(f"Batch processing error: {str(result)}")
                    results.append(error_state)
                else:
                    results.append(result)
        
        return results
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow structure."""
        return {
            "agents": [
                "DocumentPreprocessorAgent",
                "EventExtractorAgent", 
                "SubEventAnalyzerAgent",
                "KnowledgeBuilderAgent",
                "SummaryGeneratorAgent",
                "QualityEvaluatorAgent"
            ],
            "workflow_stages": [
                "preprocess",
                "extract_events", 
                "analyze_sub_events",
                "build_knowledge_graph",
                "generate_summary",
                "evaluate_quality"
            ],
            "max_retries": settings.workflow.max_retries,
            "timeout_seconds": settings.workflow.timeout_seconds,
            "parallel_processing": settings.workflow.enable_parallel_processing
        }