"""Quality Evaluator Agent for legal summary assessment."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio

from ..models.gemini_client import GeminiClient
from ..models.prompts import LegalPrompts
from ..workflow.states import LegalSummarizationState, QualityMetrics, EvaluationScores
from ..config.settings import settings

logger = logging.getLogger(__name__)


class QualityEvaluatorAgent:
    """
    Agent responsible for evaluating summary quality and providing feedback.
    Checks for legal accuracy and completeness, refines summary if needed.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the Quality Evaluator Agent."""
        self.gemini_client = gemini_client or GeminiClient()
        self.agent_type = "quality_evaluator"
        
    async def process(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """
        Evaluate summary quality and provide improvement recommendations.
        
        Args:
            state: Current workflow state with generated summary
            
        Returns:
            Updated state with evaluation scores and potential refinements
        """
        try:
            logger.info(f"Evaluating summary quality for document: {state.document_id}")
            state.update_stage("quality_evaluation", self.agent_type)
            
            if not state.generated_summary:
                state.add_error("No summary available for quality evaluation")
                return state
            
            # Perform comprehensive quality evaluation
            evaluation_scores = await self._evaluate_summary_quality(state)
            
            # Check if refinement is needed
            if self._needs_refinement(evaluation_scores):
                logger.info("Summary needs refinement, attempting to improve")
                refined_summary = await self._refine_summary(state, evaluation_scores)
                
                if refined_summary and len(refined_summary.strip()) > 0:
                    # Re-evaluate refined summary
                    state.generated_summary = refined_summary
                    evaluation_scores = await self._evaluate_summary_quality(state)
                    state.quality_feedback.append("Summary was automatically refined based on quality assessment")
            
            # Update state with evaluation results
            state.evaluation_scores = evaluation_scores
            
            logger.info(f"Quality evaluation completed. Overall score: {evaluation_scores.legal_metrics.overall_score:.2f}")
            return state
            
        except Exception as e:
            error_msg = f"Error in quality evaluation: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def _evaluate_summary_quality(self, state: LegalSummarizationState) -> EvaluationScores:
        """Perform comprehensive quality evaluation of the summary."""
        try:
            # Run multiple evaluation tasks in parallel
            tasks = [
                self._evaluate_legal_quality(state),
                self._calculate_rouge_scores(state),
                self._calculate_bert_score(state) if settings.evaluation.enable_bertscore else asyncio.coroutine(lambda: 0.0)()
            ]
            
            legal_metrics, rouge_scores, bert_score = await asyncio.gather(*tasks)
            
            # Combine all evaluation metrics
            evaluation_scores = EvaluationScores(
                rouge_1=rouge_scores.get("rouge1", 0.0),
                rouge_2=rouge_scores.get("rouge2", 0.0),
                rouge_l=rouge_scores.get("rougeL", 0.0),
                bert_score=bert_score,
                legal_metrics=legal_metrics
            )
            
            return evaluation_scores
            
        except Exception as e:
            logger.warning(f"Failed to evaluate summary quality: {str(e)}, using fallback")
            return self._evaluate_quality_fallback(state)
    
    async def _evaluate_legal_quality(self, state: LegalSummarizationState) -> QualityMetrics:
        """Evaluate legal-specific quality metrics using LLM."""
        try:
            # Prepare evaluation context
            evaluation_context = self._prepare_evaluation_context(state)
            
            prompt = LegalPrompts.format_prompt(
                LegalPrompts.QUALITY_EVALUATION,
                summary=state.generated_summary,
                original_text=state.document_text[:3000],  # First 3000 chars for context
                key_events=self._format_key_events(state.major_events)
            )
            
            system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
            
            response = await self.gemini_client.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                schema={
                    "legal_accuracy": "number",
                    "completeness": "number", 
                    "coherence": "number",
                    "clarity": "number",
                    "legal_language": "number",
                    "improvement_suggestions": ["string"]
                }
            )
            
            # Calculate overall score
            scores = [
                response.get("legal_accuracy", 5.0),
                response.get("completeness", 5.0),
                response.get("coherence", 5.0),
                response.get("clarity", 5.0),
                response.get("legal_language", 5.0)
            ]
            overall_score = sum(scores) / len(scores)
            
            return QualityMetrics(
                legal_accuracy=response.get("legal_accuracy", 5.0),
                completeness=response.get("completeness", 5.0),
                coherence=response.get("coherence", 5.0),
                clarity=response.get("clarity", 5.0),
                legal_language=response.get("legal_language", 5.0),
                overall_score=overall_score,
                improvement_suggestions=response.get("improvement_suggestions", [])
            )
            
        except Exception as e:
            logger.warning(f"Failed to evaluate legal quality with LLM: {str(e)}, using fallback")
            return self._evaluate_legal_quality_fallback(state)
    
    def _prepare_evaluation_context(self, state: LegalSummarizationState) -> str:
        """Prepare context for quality evaluation."""
        context_parts = []
        
        # Add case metadata for reference
        if state.case_metadata:
            context_parts.append(f"Case: {state.case_metadata.case_number}")
            context_parts.append(f"Court: {state.case_metadata.court}")
        
        # Add key sections for comparison
        if state.document_sections.facts:
            context_parts.append(f"Original Facts (excerpt): {state.document_sections.facts[:500]}...")
        
        if state.document_sections.conclusion:
            context_parts.append(f"Original Conclusion (excerpt): {state.document_sections.conclusion[:500]}...")
        
        return "\n\n".join(context_parts)
    
    def _format_key_events(self, events: List[Any]) -> str:
        """Format key events for evaluation context."""
        if not events:
            return "No major events identified"
        
        formatted = "Key Events:\n"
        for i, event in enumerate(events[:5], 1):
            formatted += f"{i}. {event.title} ({event.event_type})\n"
        
        return formatted
    
    async def _calculate_rouge_scores(self, state: LegalSummarizationState) -> Dict[str, float]:
        """Calculate ROUGE scores comparing summary to reference."""
        try:
            from rouge_score import rouge_scorer
            
            # Use document conclusion as reference if available
            reference_text = ""
            if state.document_sections.conclusion:
                reference_text = state.document_sections.conclusion
            elif state.document_sections.facts:
                reference_text = state.document_sections.facts
            else:
                # Use first 1000 words of document as reference
                reference_text = " ".join(state.document_text.split()[:1000])
            
            if not reference_text.strip():
                logger.warning("No reference text available for ROUGE calculation")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(
                settings.evaluation.rouge_types, 
                use_stemmer=True
            )
            
            # Calculate scores
            scores = scorer.score(reference_text, state.generated_summary)
            
            # Extract F1 scores
            rouge_scores = {}
            for rouge_type in settings.evaluation.rouge_types:
                if rouge_type in scores:
                    rouge_scores[rouge_type] = scores[rouge_type].fmeasure
            
            return rouge_scores
            
        except ImportError:
            logger.warning("rouge_score package not available, skipping ROUGE calculation")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        except Exception as e:
            logger.warning(f"Failed to calculate ROUGE scores: {str(e)}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    async def _calculate_bert_score(self, state: LegalSummarizationState) -> float:
        """Calculate BERTScore for semantic similarity."""
        try:
            from bert_score import score
            
            # Use document sections as references
            references = []
            if state.document_sections.facts:
                references.append(state.document_sections.facts)
            if state.document_sections.reasoning:
                references.append(state.document_sections.reasoning)
            if state.document_sections.conclusion:
                references.append(state.document_sections.conclusion)
            
            if not references:
                logger.warning("No reference text available for BERTScore calculation")
                return 0.0
            
            # Combine references
            reference_text = " ".join(references)
            
            # Calculate BERTScore
            P, R, F1 = score(
                [state.generated_summary], 
                [reference_text],
                model_type=settings.evaluation.bertscore_model,
                verbose=False
            )
            
            return F1.item()
            
        except ImportError:
            logger.warning("bert_score package not available, skipping BERTScore calculation")
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate BERTScore: {str(e)}")
            return 0.0
    
    def _needs_refinement(self, evaluation_scores: EvaluationScores) -> bool:
        """Determine if summary needs refinement based on evaluation scores."""
        # Define thresholds for refinement
        min_overall_score = 6.0  # Out of 10
        min_completeness = 5.5
        min_legal_accuracy = 6.0
        
        legal_metrics = evaluation_scores.legal_metrics
        
        return (
            legal_metrics.overall_score < min_overall_score or
            legal_metrics.completeness < min_completeness or
            legal_metrics.legal_accuracy < min_legal_accuracy
        )
    
    async def _refine_summary(
        self, 
        state: LegalSummarizationState, 
        evaluation_scores: EvaluationScores
    ) -> Optional[str]:
        """Refine summary based on evaluation feedback."""
        try:
            # Prepare refinement context
            key_events = self._format_key_events(state.major_events)
            legal_issues = self._format_legal_issues(state.legal_reasoning)
            
            # Create improvement suggestions based on evaluation
            improvement_areas = []
            legal_metrics = evaluation_scores.legal_metrics
            
            if legal_metrics.legal_accuracy < 6.0:
                improvement_areas.append("Improve legal accuracy and precision of legal terminology")
            
            if legal_metrics.completeness < 6.0:
                improvement_areas.append("Include all major legal points and events")
            
            if legal_metrics.coherence < 6.0:
                improvement_areas.append("Improve logical flow and structure")
            
            if legal_metrics.clarity < 6.0:
                improvement_areas.append("Enhance clarity and readability")
            
            # Add specific suggestions from evaluation
            improvement_areas.extend(legal_metrics.improvement_suggestions)
            
            prompt = LegalPrompts.format_prompt(
                LegalPrompts.SUMMARY_REFINEMENT,
                original_summary=state.generated_summary,
                key_events=key_events,
                legal_issues=legal_issues
            )
            
            # Add improvement guidance
            improvement_guidance = "\n".join([f"- {area}" for area in improvement_areas])
            prompt += f"\n\nSpecific areas for improvement:\n{improvement_guidance}"
            
            system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
            
            refined_summary = await self.gemini_client.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.2  # Lower temperature for refinement
            )
            
            # Validate that refinement is actually an improvement
            if len(refined_summary.split()) < len(state.generated_summary.split()) * 0.8:
                logger.warning("Refined summary too short, keeping original")
                return None
            
            return refined_summary.strip()
            
        except Exception as e:
            logger.warning(f"Failed to refine summary: {str(e)}")
            return None
    
    def _format_legal_issues(self, legal_reasoning: Optional[Any]) -> str:
        """Format legal issues for refinement context."""
        if not legal_reasoning or not legal_reasoning.key_issues:
            return "No specific legal issues identified"
        
        formatted = "Key Legal Issues:\n"
        for issue in legal_reasoning.key_issues:
            formatted += f"- {issue}\n"
        
        return formatted
    
    def _evaluate_legal_quality_fallback(self, state: LegalSummarizationState) -> QualityMetrics:
        """Fallback method for legal quality evaluation."""
        # Simple heuristic-based evaluation
        summary_words = len(state.generated_summary.split())
        original_words = len(state.document_text.split())
        
        # Basic metrics based on summary length and content
        if summary_words == 0:
            return QualityMetrics(
                legal_accuracy=0.0,
                completeness=0.0,
                coherence=0.0,
                clarity=0.0,
                legal_language=0.0,
                overall_score=0.0,
                improvement_suggestions=["Summary is empty"]
            )
        
        # Heuristic scoring based on various factors
        length_score = min(10.0, (summary_words / state.target_length) * 10)
        
        # Check for legal terminology
        legal_terms = [
            'court', 'judgment', 'case', 'legal', 'law', 'petition', 'appeal',
            'defendant', 'plaintiff', 'evidence', 'witness', 'precedent'
        ]
        
        summary_lower = state.generated_summary.lower()
        legal_term_count = sum(1 for term in legal_terms if term in summary_lower)
        legal_language_score = min(10.0, (legal_term_count / 5) * 10)
        
        # Basic completeness check
        completeness_score = 7.0  # Default moderate score
        if state.major_events:
            events_mentioned = sum(
                1 for event in state.major_events 
                if any(word in summary_lower for word in event.title.lower().split())
            )
            completeness_score = min(10.0, (events_mentioned / len(state.major_events)) * 10)
        
        # Default scores for other metrics
        legal_accuracy = 6.0
        coherence = 7.0
        clarity = 7.0
        
        overall_score = (
            legal_accuracy + completeness_score + coherence + 
            clarity + legal_language_score
        ) / 5
        
        return QualityMetrics(
            legal_accuracy=legal_accuracy,
            completeness=completeness_score,
            coherence=coherence,
            clarity=clarity,
            legal_language=legal_language_score,
            overall_score=overall_score,
            improvement_suggestions=[]
        )
    
    def _evaluate_quality_fallback(self, state: LegalSummarizationState) -> EvaluationScores:
        """Fallback method for complete quality evaluation."""
        legal_metrics = self._evaluate_legal_quality_fallback(state)
        
        return EvaluationScores(
            rouge_1=0.0,
            rouge_2=0.0,
            rouge_l=0.0,
            bert_score=0.0,
            legal_metrics=legal_metrics
        )