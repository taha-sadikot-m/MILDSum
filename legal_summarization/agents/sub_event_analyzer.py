"""Sub-Event Analyzer Agent for detailed legal event analysis."""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..models.gemini_client import GeminiClient
from ..models.prompts import LegalPrompts
from ..workflow.states import LegalSummarizationState, MajorEvent, SubEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SubEventAnalyzerAgent:
    """
    Agent responsible for extracting detailed sub-events for each major event.
    Identifies causal relationships and extracts legal precedents and citations.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the Sub-Event Analyzer Agent."""
        self.gemini_client = gemini_client or GeminiClient()
        self.agent_type = "sub_event_analyzer"
        
    async def process(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """
        Analyze major events to extract detailed sub-events and relationships.
        
        Args:
            state: Current workflow state with major events
            
        Returns:
            Updated state with enriched events containing sub-events
        """
        try:
            logger.info(f"Starting sub-event analysis for {len(state.major_events)} major events")
            state.update_stage("sub_event_analysis", self.agent_type)
            
            # Process each major event to extract sub-events
            enriched_events = []
            
            # Process events in parallel batches to optimize performance
            batch_size = 3
            for i in range(0, len(state.major_events), batch_size):
                batch = state.major_events[i:i + batch_size]
                
                # Process batch in parallel
                tasks = [
                    self._analyze_major_event(event, state.document_text) 
                    for event in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to analyze event: {str(result)}")
                        continue
                    enriched_events.append(result)
            
            # Update state with enriched events
            state.major_events = enriched_events
            
            logger.info(f"Sub-event analysis completed for {len(enriched_events)} events")
            return state
            
        except Exception as e:
            error_msg = f"Error in sub-event analysis: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def _analyze_major_event(
        self, 
        major_event: MajorEvent, 
        document_text: str
    ) -> MajorEvent:
        """Analyze a single major event to extract sub-events and details."""
        try:
            # Find relevant context for this event in the document
            context = self._extract_event_context(major_event, document_text)
            
            # Extract sub-events using LLM
            sub_events = await self._extract_sub_events(major_event, context)
            
            # Create enriched major event
            enriched_event = MajorEvent(
                event_id=major_event.event_id,
                title=major_event.title,
                description=major_event.description,
                timestamp=major_event.timestamp,
                legal_significance=major_event.legal_significance,
                event_type=major_event.event_type,
                sub_events=sub_events
            )
            
            return enriched_event
            
        except Exception as e:
            logger.warning(f"Failed to analyze event {major_event.event_id}: {str(e)}")
            # Return original event if analysis fails
            return major_event
    
    def _extract_event_context(self, major_event: MajorEvent, document_text: str) -> str:
        """Extract relevant context for the major event from the document."""
        # Simple approach: find text surrounding event description
        description_start = document_text.find(major_event.description[:100])
        
        if description_start != -1:
            # Extract context around the event (±1000 characters)
            start_pos = max(0, description_start - 1000)
            end_pos = min(len(document_text), description_start + len(major_event.description) + 1000)
            context = document_text[start_pos:end_pos]
        else:
            # If exact match not found, use event description and search for related content
            context = self._find_related_content(major_event, document_text)
        
        return context
    
    def _find_related_content(self, major_event: MajorEvent, document_text: str) -> str:
        """Find content related to the major event using keyword matching."""
        # Extract keywords from event title and description
        import re
        
        text_to_analyze = f"{major_event.title} {major_event.description}"
        keywords = re.findall(r'\b[A-Za-z]{4,}\b', text_to_analyze.lower())
        
        # Remove common stop words
        stop_words = {
            'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'been',
            'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could'
        }
        keywords = [k for k in keywords if k not in stop_words]
        
        # Find paragraphs containing these keywords
        paragraphs = document_text.split('\n')
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            score = sum(1 for keyword in keywords if keyword in paragraph_lower)
            
            if score >= 2:  # Require at least 2 keyword matches
                relevant_paragraphs.append(paragraph)
        
        # Return top 5 most relevant paragraphs
        return '\n'.join(relevant_paragraphs[:5])
    
    async def _extract_sub_events(
        self, 
        major_event: MajorEvent, 
        context: str
    ) -> List[SubEvent]:
        """Extract sub-events for a major event using Gemini LLM."""
        try:
            prompt = LegalPrompts.format_prompt(
                LegalPrompts.SUB_EVENT_ANALYSIS,
                major_event=f"Title: {major_event.title}\nDescription: {major_event.description}",
                context=context
            )
            
            system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
            
            response = await self.gemini_client.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                schema={
                    "sub_events": [
                        {
                            "sub_event_id": "string",
                            "description": "string",
                            "evidence": ["string"],
                            "legal_references": ["string"],
                            "timestamp": "string",
                            "participants": ["string"]
                        }
                    ]
                }
            )
            
            sub_events_data = response.get("sub_events", [])
            sub_events = []
            
            for i, sub_event_data in enumerate(sub_events_data):
                sub_event = SubEvent(
                    sub_event_id=f"{major_event.event_id}_sub_{i}",
                    description=sub_event_data.get("description", ""),
                    evidence=sub_event_data.get("evidence", []),
                    legal_references=sub_event_data.get("legal_references", []),
                    timestamp=sub_event_data.get("timestamp", ""),
                    participants=sub_event_data.get("participants", [])
                )
                sub_events.append(sub_event)
            
            return sub_events
            
        except Exception as e:
            logger.warning(f"Failed to extract sub-events with LLM: {str(e)}, using fallback")
            return self._extract_sub_events_fallback(major_event, context)
    
    def _extract_sub_events_fallback(
        self, 
        major_event: MajorEvent, 
        context: str
    ) -> List[SubEvent]:
        """Fallback method to extract sub-events using pattern matching."""
        sub_events = []
        
        if not context.strip():
            return sub_events
        
        # Split context into sentences
        import re
        sentences = re.split(r'[.!?]+', context)
        
        # Patterns that indicate sub-events
        sub_event_patterns = [
            r'(.*?(?:stated|argued|contended|submitted|claimed).*?)',
            r'(.*?(?:evidence|document|exhibit|record).*?)',
            r'(.*?(?:witness|testimony|statement).*?)',
            r'(.*?(?:precedent|citation|reference|case law).*?)',
            r'(.*?(?:procedure|process|step|action).*?)'
        ]
        
        sub_event_id = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:  # Skip very short sentences
                continue
            
            for pattern in sub_event_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Extract potential evidence/references
                    evidence = self._extract_evidence_references(sentence)
                    legal_refs = self._extract_legal_references(sentence)
                    
                    sub_event = SubEvent(
                        sub_event_id=f"{major_event.event_id}_fallback_sub_{sub_event_id}",
                        description=sentence,
                        evidence=evidence,
                        legal_references=legal_refs,
                        timestamp="",
                        participants=[]
                    )
                    sub_events.append(sub_event)
                    sub_event_id += 1
                    break
        
        # Limit number of sub-events
        return sub_events[:10]
    
    def _extract_evidence_references(self, text: str) -> List[str]:
        """Extract evidence references from text."""
        import re
        evidence = []
        
        # Patterns for evidence
        evidence_patterns = [
            r'(exhibit\s+[A-Z0-9]+)',
            r'(document\s+[A-Z0-9]+)',
            r'(affidavit\s+(?:of\s+)?[A-Za-z\s]+)',
            r'(record\s+[A-Z0-9]+)',
            r'(statement\s+(?:of\s+)?[A-Za-z\s]+)'
        ]
        
        for pattern in evidence_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            evidence.extend(matches)
        
        return evidence[:5]  # Limit to 5 evidence items
    
    def _extract_legal_references(self, text: str) -> List[str]:
        """Extract legal references and citations from text."""
        import re
        references = []
        
        # Patterns for legal references
        legal_patterns = [
            r'((?:section|article|rule)\s+\d+[A-Za-z]*(?:\s*\([^)]+\))?)',
            r'((?:[A-Z][a-z]+\s+)+v\.?\s+(?:[A-Z][a-z]+\s*)+)',  # Case names
            r'(\d{4}\s+[A-Z]+\s+\d+)',  # Citation format like "2020 SCR 123"
            r'(AIR\s+\d{4}\s+[A-Z]+\s+\d+)',  # AIR citations
            r'((?:Act|Code|Rules?)\s+\d{4})'  # Act references
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return references[:5]  # Limit to 5 legal references