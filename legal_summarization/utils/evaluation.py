"""Evaluation utilities for legal summarization."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..workflow.states import LegalSummarizationState, EvaluationScores

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bert_score: float
    legal_accuracy: float
    completeness: float
    coherence: float
    clarity: float
    overall_score: float


class LegalSummarizationEvaluator:
    """Comprehensive evaluator for legal summarization quality."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.rouge_scorer = None
        self.bert_scorer = None
        self._initialize_scorers()
    
    def _initialize_scorers(self):
        """Initialize scoring modules."""
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        except ImportError:
            logger.warning("ROUGE scorer not available")
        
        try:
            import bert_score
            self.bert_scorer = bert_score
        except ImportError:
            logger.warning("BERTScore not available")
    
    def evaluate_summary(
        self, 
        generated_summary: str, 
        reference_text: str,
        original_document: str = "",
        legal_events: List[Any] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of generated summary.
        
        Args:
            generated_summary: Generated summary text
            reference_text: Reference summary or key sections
            original_document: Original legal document
            legal_events: Extracted legal events for completeness check
            
        Returns:
            Comprehensive evaluation metrics
        """
        try:
            # Calculate ROUGE scores
            rouge_scores = self._calculate_rouge_scores(generated_summary, reference_text)
            
            # Calculate BERTScore
            bert_score = self._calculate_bert_score(generated_summary, reference_text)
            
            # Calculate legal-specific metrics
            legal_metrics = self._calculate_legal_metrics(
                generated_summary, reference_text, original_document, legal_events
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(rouge_scores, bert_score, legal_metrics)
            
            return EvaluationMetrics(
                rouge_1=rouge_scores.get('rouge1', 0.0),
                rouge_2=rouge_scores.get('rouge2', 0.0),
                rouge_l=rouge_scores.get('rougeL', 0.0),
                bert_score=bert_score,
                legal_accuracy=legal_metrics['accuracy'],
                completeness=legal_metrics['completeness'],
                coherence=legal_metrics['coherence'],
                clarity=legal_metrics['clarity'],
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not self.rouge_scorer or not generated.strip() or not reference.strip():
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def _calculate_bert_score(self, generated: str, reference: str) -> float:
        """Calculate BERTScore."""
        if not self.bert_scorer or not generated.strip() or not reference.strip():
            return 0.0
        
        try:
            P, R, F1 = self.bert_scorer.score([generated], [reference], verbose=False)
            return F1.item()
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_legal_metrics(
        self, 
        generated: str, 
        reference: str, 
        original: str,
        events: List[Any] = None
    ) -> Dict[str, float]:
        """Calculate legal domain-specific metrics."""
        try:
            # Legal accuracy - check for legal terminology preservation
            accuracy = self._assess_legal_accuracy(generated, reference, original)
            
            # Completeness - check coverage of key events and facts
            completeness = self._assess_completeness(generated, events, original)
            
            # Coherence - check logical flow and structure
            coherence = self._assess_coherence(generated)
            
            # Clarity - check readability and understandability
            clarity = self._assess_clarity(generated)
            
            return {
                'accuracy': accuracy,
                'completeness': completeness,
                'coherence': coherence,
                'clarity': clarity
            }
            
        except Exception as e:
            logger.warning(f"Legal metrics calculation failed: {str(e)}")
            return {'accuracy': 5.0, 'completeness': 5.0, 'coherence': 5.0, 'clarity': 5.0}
    
    def _assess_legal_accuracy(self, generated: str, reference: str, original: str) -> float:
        """Assess legal accuracy based on terminology and fact preservation."""
        if not generated.strip():
            return 0.0
        
        # Define important legal terms
        legal_terms = [
            'court', 'judgment', 'petition', 'appeal', 'defendant', 'plaintiff',
            'evidence', 'witness', 'precedent', 'law', 'statute', 'constitutional',
            'legal', 'ruling', 'order', 'case', 'jurisdiction', 'liability'
        ]
        
        # Check presence of legal terminology
        generated_lower = generated.lower()
        reference_lower = reference.lower() if reference else ""
        original_lower = original.lower() if original else ""
        
        # Calculate term preservation ratio
        reference_terms = set()
        for term in legal_terms:
            if term in reference_lower or term in original_lower:
                reference_terms.add(term)
        
        preserved_terms = set()
        for term in reference_terms:
            if term in generated_lower:
                preserved_terms.add(term)
        
        if reference_terms:
            term_preservation = len(preserved_terms) / len(reference_terms)
        else:
            term_preservation = 0.5  # Default score if no reference terms
        
        # Check for factual consistency (simple heuristic)
        factual_consistency = self._check_factual_consistency(generated, reference, original)
        
        # Combine scores (scale to 0-10)
        accuracy_score = (term_preservation * 0.6 + factual_consistency * 0.4) * 10
        return min(10.0, max(0.0, accuracy_score))
    
    def _check_factual_consistency(self, generated: str, reference: str, original: str) -> float:
        """Check factual consistency using simple pattern matching."""
        import re
        
        # Extract potential facts (dates, numbers, names)
        fact_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',  # Dates
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Case names
            r'\b(?:Section|Article|Rule)\s+\d+\b',  # Legal references
        ]
        
        # Find facts in reference/original
        reference_facts = set()
        for pattern in fact_patterns:
            if reference:
                reference_facts.update(re.findall(pattern, reference))
            if original:
                reference_facts.update(re.findall(pattern, original[:2000]))  # First 2000 chars
        
        # Find facts in generated summary
        generated_facts = set()
        for pattern in fact_patterns:
            generated_facts.update(re.findall(pattern, generated))
        
        # Calculate consistency
        if reference_facts:
            consistent_facts = reference_facts.intersection(generated_facts)
            consistency = len(consistent_facts) / len(reference_facts)
        else:
            consistency = 0.8  # Default high score if no factual patterns found
        
        return consistency
    
    def _assess_completeness(self, generated: str, events: List[Any], original: str) -> float:
        """Assess completeness based on coverage of key events and concepts."""
        if not generated.strip():
            return 0.0
        
        generated_lower = generated.lower()
        
        # Check event coverage if events are provided
        event_coverage = 0.0
        if events:
            covered_events = 0
            for event in events:
                event_keywords = event.title.lower().split() if hasattr(event, 'title') else []
                if any(keyword in generated_lower for keyword in event_keywords if len(keyword) > 3):
                    covered_events += 1
            
            event_coverage = covered_events / len(events) if events else 0.0
        
        # Check coverage of key legal concepts
        key_concepts = [
            'facts', 'issue', 'legal', 'court', 'decision', 'ruling', 
            'evidence', 'law', 'case', 'judgment', 'conclusion'
        ]
        
        concept_coverage = 0.0
        covered_concepts = sum(1 for concept in key_concepts if concept in generated_lower)
        concept_coverage = covered_concepts / len(key_concepts)
        
        # Check length appropriateness
        word_count = len(generated.split())
        length_score = 1.0
        if word_count < 100:
            length_score = word_count / 100  # Penalize very short summaries
        elif word_count > 1000:
            length_score = max(0.5, 1000 / word_count)  # Penalize very long summaries
        
        # Combine scores (scale to 0-10)
        completeness_score = (event_coverage * 0.4 + concept_coverage * 0.4 + length_score * 0.2) * 10
        return min(10.0, max(0.0, completeness_score))
    
    def _assess_coherence(self, generated: str) -> float:
        """Assess logical flow and coherence of the summary."""
        if not generated.strip():
            return 0.0
        
        sentences = generated.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 5.0  # Default score for very short summaries
        
        # Check for transition words and logical connectors
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'accordingly', 'subsequently', 'nevertheless', 'additionally',
            'finally', 'thus', 'hence', 'whereas', 'although'
        ]
        
        transition_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                transition_count += 1
        
        # Check for proper paragraph structure (heuristic)
        structure_score = 0.8  # Default good structure score
        
        # Check sentence length variation (indicates good writing)
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            if length_variance > 50:  # Good variation
                structure_score += 0.1
        
        # Combine scores (scale to 0-10)
        transition_score = min(1.0, transition_count / max(1, len(sentences) * 0.3))
        coherence_score = (transition_score * 0.5 + structure_score * 0.5) * 10
        return min(10.0, max(0.0, coherence_score))
    
    def _assess_clarity(self, generated: str) -> float:
        """Assess clarity and readability of the summary."""
        if not generated.strip():
            return 0.0
        
        words = generated.split()
        sentences = generated.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal sentence length for legal writing (15-25 words)
        length_score = 1.0
        if avg_sentence_length < 10:
            length_score = avg_sentence_length / 10
        elif avg_sentence_length > 30:
            length_score = max(0.3, 30 / avg_sentence_length)
        
        # Check for overly complex words (simple heuristic)
        complex_words = [word for word in words if len(word) > 10]
        complexity_ratio = len(complex_words) / len(words)
        complexity_score = max(0.3, 1.0 - complexity_ratio * 2)  # Penalize excessive complexity
        
        # Check for legal jargon balance
        legal_terms = ['whereas', 'heretofore', 'aforementioned', 'pursuant', 'notwithstanding']
        jargon_count = sum(1 for word in words if word.lower() in legal_terms)
        jargon_ratio = jargon_count / len(words)
        jargon_score = 1.0 if jargon_ratio < 0.05 else max(0.5, 1.0 - jargon_ratio * 10)
        
        # Combine scores (scale to 0-10)
        clarity_score = (length_score * 0.4 + complexity_score * 0.3 + jargon_score * 0.3) * 10
        return min(10.0, max(0.0, clarity_score))
    
    def _calculate_overall_score(
        self, 
        rouge_scores: Dict[str, float], 
        bert_score: float, 
        legal_metrics: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score."""
        # Weight different metrics appropriately for legal summarization
        weights = {
            'rouge_2': 0.15,  # ROUGE-2 for bigram overlap
            'rouge_l': 0.15,  # ROUGE-L for longest common subsequence
            'bert_score': 0.20,  # Semantic similarity
            'legal_accuracy': 0.25,  # Most important for legal domain
            'completeness': 0.15,  # Coverage of key points
            'coherence': 0.05,  # Logical flow
            'clarity': 0.05   # Readability
        }
        
        score_components = {
            'rouge_2': rouge_scores.get('rouge2', 0.0) * 10,  # Scale to 0-10
            'rouge_l': rouge_scores.get('rougeL', 0.0) * 10,
            'bert_score': bert_score * 10,
            'legal_accuracy': legal_metrics['accuracy'],
            'completeness': legal_metrics['completeness'],
            'coherence': legal_metrics['coherence'],
            'clarity': legal_metrics['clarity']
        }
        
        # Calculate weighted average
        overall_score = sum(
            score_components[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return min(10.0, max(0.0, overall_score))
    
    def evaluate_batch(
        self, 
        results: List[LegalSummarizationState],
        references: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of summarization results.
        
        Args:
            results: List of workflow states
            references: Optional list of reference summaries
            
        Returns:
            Batch evaluation metrics
        """
        try:
            batch_metrics = {
                'total_documents': len(results),
                'successful_summaries': 0,
                'average_scores': {},
                'score_distribution': {},
                'individual_results': []
            }
            
            all_scores = []
            metric_sums = {
                'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0,
                'bert_score': 0.0, 'legal_accuracy': 0.0,
                'completeness': 0.0, 'coherence': 0.0, 'clarity': 0.0, 'overall_score': 0.0
            }
            
            for i, result in enumerate(results):
                if not result.generated_summary:
                    continue
                
                # Use reference if available, otherwise use document sections
                reference = ""
                if references and i < len(references):
                    reference = references[i]
                elif result.document_sections:
                    reference = f"{result.document_sections.facts} {result.document_sections.conclusion}"
                
                # Evaluate this result
                metrics = self.evaluate_summary(
                    result.generated_summary,
                    reference,
                    result.document_text,
                    result.major_events
                )
                
                all_scores.append(metrics)
                batch_metrics['successful_summaries'] += 1
                
                # Add to sums for averaging
                for attr in metric_sums.keys():
                    metric_sums[attr] += getattr(metrics, attr)
                
                # Store individual result
                batch_metrics['individual_results'].append({
                    'document_id': result.document_id,
                    'metrics': metrics
                })
            
            # Calculate averages
            if batch_metrics['successful_summaries'] > 0:
                for metric in metric_sums.keys():
                    batch_metrics['average_scores'][metric] = (
                        metric_sums[metric] / batch_metrics['successful_summaries']
                    )
            
            # Calculate score distributions
            if all_scores:
                overall_scores = [s.overall_score for s in all_scores]
                batch_metrics['score_distribution'] = {
                    'min': min(overall_scores),
                    'max': max(overall_scores),
                    'median': np.median(overall_scores),
                    'std_dev': np.std(overall_scores)
                }
            
            return batch_metrics
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {str(e)}")
            return {'error': str(e)}


# Helper functions for specific evaluation tasks
def calculate_summary_statistics(state: LegalSummarizationState) -> Dict[str, Any]:
    """Calculate basic statistics about the summarization result."""
    stats = {
        'document_word_count': len(state.document_text.split()) if state.document_text else 0,
        'summary_word_count': len(state.generated_summary.split()) if state.generated_summary else 0,
        'compression_ratio': 0.0,
        'events_extracted': len(state.major_events),
        'sub_events_total': sum(len(event.sub_events) for event in state.major_events),
        'legal_references_found': 0,
        'processing_errors': len(state.error_messages),
        'retry_count': state.retry_count
    }
    
    # Calculate compression ratio
    if stats['document_word_count'] > 0:
        stats['compression_ratio'] = stats['summary_word_count'] / stats['document_word_count']
    
    # Count legal references
    for event in state.major_events:
        for sub_event in event.sub_events:
            stats['legal_references_found'] += len(sub_event.legal_references)
    
    return stats