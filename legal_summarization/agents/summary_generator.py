"""Summary Generator Agent for legal document summarization."""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..models.gemini_client import GeminiClient
from ..models.prompts import LegalPrompts
from ..workflow.states import LegalSummarizationState, SummaryComponents
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SummaryGeneratorAgent:
    """
    Agent responsible for generating coherent abstractive summaries.
    Uses extracted events to create high-quality legal summaries.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the Summary Generator Agent."""
        self.gemini_client = gemini_client or GeminiClient()
        self.agent_type = "summary_generator"
        
    async def process(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """
        Generate comprehensive legal summary from extracted events and analysis.
        
        Args:
            state: Current workflow state with events and knowledge graph
            
        Returns:
            Updated state with generated summary
        """
        try:
            logger.info(f"Generating legal summary for document: {state.document_id}")
            state.update_stage("summary_generation", self.agent_type)
            
            # Generate summary components first
            summary_components = await self._generate_summary_components(state)
            
            # Generate final coherent summary
            final_summary = await self._generate_final_summary(state, summary_components)
            
            # Update state
            state.summary_components = summary_components
            state.generated_summary = final_summary
            
            logger.info(f"Generated summary of {len(final_summary.split())} words")
            return state
            
        except Exception as e:
            error_msg = f"Error in summary generation: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def _generate_summary_components(self, state: LegalSummarizationState) -> SummaryComponents:
        """Generate individual components of the summary."""
        try:
            # Generate components in parallel for efficiency
            tasks = [
                self._generate_facts_summary(state),
                self._generate_issues_summary(state),
                self._generate_reasoning_summary(state),
                self._generate_conclusion_summary(state)
            ]
            
            facts, issues, reasoning, conclusion = await asyncio.gather(*tasks)
            
            return SummaryComponents(
                facts=facts,
                issues=issues,
                reasoning=reasoning,
                conclusion=conclusion
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate summary components: {str(e)}, using fallback")
            return self._generate_summary_components_fallback(state)
    
    async def _generate_facts_summary(self, state: LegalSummarizationState) -> str:
        """Generate summary of factual background."""
        # Prepare facts context from document sections and events
        facts_context = self._prepare_facts_context(state)
        
        prompt = f"""
Summarize the factual background of this legal case in a clear, concise manner.
Focus on the key facts that are legally relevant to the case outcome.

Facts Context:
{facts_context}

Generate a factual summary that:
1. Presents facts chronologically where relevant
2. Includes all parties involved
3. Describes the circumstances leading to the legal dispute
4. Maintains objectivity and legal precision
5. Is approximately 100-150 words

Provide only the factual summary, no additional commentary.
"""
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        return await self.gemini_client.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )
    
    async def _generate_issues_summary(self, state: LegalSummarizationState) -> str:
        """Generate summary of legal issues."""
        # Prepare issues context
        issues_context = self._prepare_issues_context(state)
        
        prompt = f"""
Identify and summarize the key legal issues that the court needed to resolve in this case.

Legal Issues Context:
{issues_context}

Generate a summary of legal issues that:
1. Lists the main legal questions
2. Explains why each issue is legally significant
3. Shows the relationship between different issues
4. Uses precise legal terminology
5. Is approximately 80-120 words

Provide only the issues summary.
"""
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        return await self.gemini_client.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )
    
    async def _generate_reasoning_summary(self, state: LegalSummarizationState) -> str:
        """Generate summary of court's legal reasoning."""
        reasoning_context = self._prepare_reasoning_context(state)
        
        prompt = f"""
Summarize the court's legal reasoning and analysis in this case.

Legal Reasoning Context:
{reasoning_context}

Generate a reasoning summary that:
1. Explains the court's legal analysis step by step
2. References key precedents and legal principles applied
3. Shows how the court reached its conclusions
4. Maintains the logical flow of legal reasoning
5. Is approximately 150-200 words

Provide only the reasoning summary.
"""
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        return await self.gemini_client.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )
    
    async def _generate_conclusion_summary(self, state: LegalSummarizationState) -> str:
        """Generate summary of the court's conclusion and orders."""
        conclusion_context = self._prepare_conclusion_context(state)
        
        prompt = f"""
Summarize the court's final conclusion, judgment, and any orders issued.

Conclusion Context:
{conclusion_context}

Generate a conclusion summary that:
1. States the court's final decision clearly
2. Explains the outcome for each party
3. Lists any specific orders or directions given
4. Mentions any costs or remedies awarded
5. Is approximately 80-120 words

Provide only the conclusion summary.
"""
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        return await self.gemini_client.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )
    
    def _prepare_facts_context(self, state: LegalSummarizationState) -> str:
        """Prepare context for facts summary generation."""
        context_parts = []
        
        # Add document facts section
        if state.document_sections.facts:
            context_parts.append(f"Document Facts Section:\n{state.document_sections.facts}")
        
        # Add case metadata
        if state.case_metadata:
            metadata_context = f"""
Case Details:
- Case Number: {state.case_metadata.case_number}
- Court: {state.case_metadata.court}
- Parties: {', '.join(state.case_metadata.parties)}
- Date: {state.case_metadata.date}
"""
            context_parts.append(metadata_context)
        
        # Add relevant procedural events
        procedural_events = [e for e in state.major_events if e.event_type == "procedural"]
        if procedural_events:
            events_context = "Key Procedural Events:\n"
            for event in procedural_events[:3]:
                events_context += f"- {event.title}: {event.description[:200]}\n"
            context_parts.append(events_context)
        
        return "\n\n".join(context_parts)
    
    def _prepare_issues_context(self, state: LegalSummarizationState) -> str:
        """Prepare context for legal issues summary."""
        context_parts = []
        
        # Add document legal issues section
        if state.document_sections.legal_issues:
            context_parts.append(f"Legal Issues from Document:\n{state.document_sections.legal_issues}")
        
        # Add extracted legal reasoning
        if state.legal_reasoning and state.legal_reasoning.key_issues:
            issues_context = "Identified Key Issues:\n"
            for issue in state.legal_reasoning.key_issues:
                issues_context += f"- {issue}\n"
            context_parts.append(issues_context)
        
        # Add relevant substantive events
        substantive_events = [e for e in state.major_events if e.event_type == "substantive"]
        if substantive_events:
            events_context = "Substantive Legal Events:\n"
            for event in substantive_events[:3]:
                events_context += f"- {event.title}: {event.legal_significance}\n"
            context_parts.append(events_context)
        
        return "\n\n".join(context_parts)
    
    def _prepare_reasoning_context(self, state: LegalSummarizationState) -> str:
        """Prepare context for legal reasoning summary."""
        context_parts = []
        
        # Add document reasoning section
        if state.document_sections.reasoning:
            context_parts.append(f"Court's Reasoning:\n{state.document_sections.reasoning}")
        
        # Add legal reasoning components
        if state.legal_reasoning:
            reasoning_context = "Legal Analysis Components:\n"
            
            if state.legal_reasoning.legal_principles:
                reasoning_context += "Legal Principles Applied:\n"
                for principle in state.legal_reasoning.legal_principles[:5]:
                    reasoning_context += f"- {principle}\n"
            
            if state.legal_reasoning.precedents_cited:
                reasoning_context += "\nPrecedents Cited:\n"
                for precedent in state.legal_reasoning.precedents_cited[:5]:
                    reasoning_context += f"- {precedent}\n"
            
            context_parts.append(reasoning_context)
        
        # Add major events with legal significance
        significant_events = [e for e in state.major_events if e.legal_significance]
        if significant_events:
            events_context = "Key Legal Developments:\n"
            for event in significant_events[:3]:
                events_context += f"- {event.title}: {event.legal_significance}\n"
            context_parts.append(events_context)
        
        return "\n\n".join(context_parts)
    
    def _prepare_conclusion_context(self, state: LegalSummarizationState) -> str:
        """Prepare context for conclusion summary."""
        context_parts = []
        
        # Add document conclusion section
        if state.document_sections.conclusion:
            context_parts.append(f"Court's Conclusion:\n{state.document_sections.conclusion}")
        
        # Add final events (likely to contain judgment/orders)
        final_events = state.major_events[-3:] if state.major_events else []
        if final_events:
            events_context = "Final Legal Events:\n"
            for event in final_events:
                events_context += f"- {event.title}: {event.description[:200]}\n"
            context_parts.append(events_context)
        
        return "\n\n".join(context_parts)
    
    async def _generate_final_summary(
        self, 
        state: LegalSummarizationState, 
        components: SummaryComponents
    ) -> str:
        """Generate final coherent summary from components."""
        # Prepare comprehensive context
        metadata_summary = self._prepare_metadata_summary(state)
        events_summary = self._prepare_events_summary(state)
        
        prompt = LegalPrompts.format_prompt(
            LegalPrompts.SUMMARY_GENERATION,
            metadata=metadata_summary,
            major_events=events_summary,
            legal_reasoning=self._format_legal_reasoning(state.legal_reasoning),
            target_length=state.target_length
        )
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        try:
            final_summary = await self.gemini_client.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.3
            )
            
            # Post-process summary
            final_summary = self._post_process_summary(final_summary, state.target_length)
            
            return final_summary
            
        except Exception as e:
            logger.warning(f"Failed to generate final summary with LLM: {str(e)}, using components")
            return self._generate_final_summary_fallback(components)
    
    def _prepare_metadata_summary(self, state: LegalSummarizationState) -> str:
        """Prepare metadata summary for final generation."""
        if not state.case_metadata:
            return "Case metadata not available"
        
        metadata = state.case_metadata
        return f"""
Case: {metadata.case_number}
Court: {metadata.court}
Date: {metadata.date}
Parties: {', '.join(metadata.parties)}
Case Type: {metadata.case_type}
"""
    
    def _prepare_events_summary(self, state: LegalSummarizationState) -> str:
        """Prepare events summary for final generation."""
        if not state.major_events:
            return "No major events identified"
        
        events_summary = "Major Legal Events:\n"
        for i, event in enumerate(state.major_events[:10], 1):
            events_summary += f"{i}. {event.title}\n"
            events_summary += f"   Type: {event.event_type}\n"
            events_summary += f"   Significance: {event.legal_significance}\n"
            if event.sub_events:
                events_summary += f"   Sub-events: {len(event.sub_events)} detailed components\n"
            events_summary += "\n"
        
        return events_summary
    
    def _format_legal_reasoning(self, reasoning: Optional[Any]) -> str:
        """Format legal reasoning for summary generation."""
        if not reasoning:
            return "Legal reasoning not extracted"
        
        formatted = "Legal Analysis:\n"
        
        if reasoning.key_issues:
            formatted += "Key Issues:\n"
            for issue in reasoning.key_issues:
                formatted += f"- {issue}\n"
            formatted += "\n"
        
        if reasoning.legal_principles:
            formatted += "Legal Principles:\n"
            for principle in reasoning.legal_principles:
                formatted += f"- {principle}\n"
            formatted += "\n"
        
        if reasoning.precedents_cited:
            formatted += "Precedents Cited:\n"
            for precedent in reasoning.precedents_cited:
                formatted += f"- {precedent}\n"
        
        return formatted
    
    def _post_process_summary(self, summary: str, target_length: int) -> str:
        """Post-process the generated summary for quality and length."""
        # Remove any leading/trailing whitespace
        summary = summary.strip()
        
        # Ensure proper sentence endings
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Check length and trim if necessary
        words = summary.split()
        if len(words) > target_length * 1.2:  # 20% tolerance
            # Truncate to target length, ensuring we end at a sentence boundary
            truncated_words = words[:target_length]
            truncated_text = ' '.join(truncated_words)
            
            # Find last complete sentence
            last_sentence_end = max(
                truncated_text.rfind('.'),
                truncated_text.rfind('!'),
                truncated_text.rfind('?')
            )
            
            if last_sentence_end > len(truncated_text) * 0.8:  # If we can keep 80% of content
                summary = truncated_text[:last_sentence_end + 1]
        
        return summary
    
    def _generate_summary_components_fallback(self, state: LegalSummarizationState) -> SummaryComponents:
        """Fallback method to generate summary components."""
        # Use document sections directly if available
        components = SummaryComponents()
        
        if state.document_sections.facts:
            components.facts = state.document_sections.facts[:500]  # Truncate if too long
        
        if state.document_sections.legal_issues:
            components.issues = state.document_sections.legal_issues[:400]
        
        if state.document_sections.reasoning:
            components.reasoning = state.document_sections.reasoning[:600]
        
        if state.document_sections.conclusion:
            components.conclusion = state.document_sections.conclusion[:400]
        
        return components
    
    def _generate_final_summary_fallback(self, components: SummaryComponents) -> str:
        """Fallback method to generate final summary from components."""
        summary_parts = []
        
        if components.facts:
            summary_parts.append(f"Facts: {components.facts}")
        
        if components.issues:
            summary_parts.append(f"Legal Issues: {components.issues}")
        
        if components.reasoning:
            summary_parts.append(f"Court's Reasoning: {components.reasoning}")
        
        if components.conclusion:
            summary_parts.append(f"Conclusion: {components.conclusion}")
        
        return " ".join(summary_parts)