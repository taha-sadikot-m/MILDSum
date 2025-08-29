"""Major Event Extractor Agent for legal document analysis."""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..models.gemini_client import GeminiClient
from ..models.prompts import LegalPrompts
from ..workflow.states import LegalSummarizationState, MajorEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)


class EventExtractorAgent:
    """
    Agent responsible for identifying primary legal events and proceedings.
    Extracts chronological sequence of major case developments.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the Event Extractor Agent."""
        self.gemini_client = gemini_client or GeminiClient()
        self.agent_type = "event_extractor"
        
    async def process(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """
        Extract major legal events from the document.
        
        Args:
            state: Current workflow state with preprocessed document
            
        Returns:
            Updated state with extracted major events
        """
        try:
            logger.info(f"Starting major event extraction for document: {state.document_id}")
            state.update_stage("event_extraction", self.agent_type)
            
            # Extract major events from the full document
            major_events = await self._extract_major_events(state.document_text)
            
            # Filter and rank events by legal significance
            significant_events = self._filter_significant_events(major_events)
            
            # Create chronological timeline
            timeline = self._create_timeline(significant_events)
            
            # Update state
            state.major_events = significant_events
            state.event_timeline = timeline
            
            logger.info(f"Extracted {len(significant_events)} major legal events")
            return state
            
        except Exception as e:
            error_msg = f"Error in event extraction: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def _extract_major_events(self, document_text: str) -> List[MajorEvent]:
        """Extract major events using Gemini LLM."""
        # Split document into manageable chunks if too long
        max_chunk_size = 8000  # chars
        chunks = self._split_document(document_text, max_chunk_size)
        
        all_events = []
        
        for i, chunk in enumerate(chunks):
            try:
                prompt = LegalPrompts.format_prompt(
                    LegalPrompts.MAJOR_EVENT_EXTRACTION,
                    document_text=chunk
                )
                
                system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
                
                response = await self.gemini_client.generate_structured_output(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    schema={
                        "events": [
                            {
                                "event_id": "string",
                                "title": "string", 
                                "description": "string",
                                "timestamp": "string",
                                "legal_significance": "string",
                                "event_type": "string"
                            }
                        ]
                    }
                )
                
                chunk_events = response.get("events", [])
                
                # Convert to MajorEvent objects
                for event_data in chunk_events:
                    event = MajorEvent(
                        event_id=f"event_{i}_{event_data.get('event_id', len(all_events))}",
                        title=event_data.get('title', ''),
                        description=event_data.get('description', ''),
                        timestamp=event_data.get('timestamp', ''),
                        legal_significance=event_data.get('legal_significance', ''),
                        event_type=event_data.get('event_type', 'general')
                    )
                    all_events.append(event)
                    
            except Exception as e:
                logger.warning(f"Failed to extract events from chunk {i}: {str(e)}")
                # Continue with other chunks
                continue
        
        # If LLM extraction failed completely, use fallback
        if not all_events:
            logger.warning("LLM event extraction failed, using fallback method")
            all_events = self._extract_events_fallback(document_text)
        
        return all_events
    
    def _split_document(self, text: str, max_size: int) -> List[str]:
        """Split document into manageable chunks while preserving context."""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_events_fallback(self, document_text: str) -> List[MajorEvent]:
        """Fallback method to extract events using pattern matching."""
        events = []
        
        # Common legal event indicators
        event_patterns = [
            (r'petition\s+(?:was\s+)?filed', 'procedural', 'Petition filing'),
            (r'notice\s+(?:was\s+)?issued', 'procedural', 'Notice issuance'), 
            (r'hearing\s+(?:was\s+)?held', 'procedural', 'Court hearing'),
            (r'argument(?:s)?\s+(?:were\s+)?presented', 'procedural', 'Arguments presented'),
            (r'evidence\s+(?:was\s+)?(?:presented|adduced)', 'evidential', 'Evidence presentation'),
            (r'witness(?:es)?\s+(?:were\s+)?examined', 'evidential', 'Witness examination'),
            (r'judgment\s+(?:was\s+)?(?:delivered|pronounced)', 'substantive', 'Judgment delivery'),
            (r'order\s+(?:was\s+)?passed', 'substantive', 'Order issuance'),
            (r'appeal\s+(?:was\s+)?filed', 'procedural', 'Appeal filing'),
            (r'decision\s+(?:was\s+)?(?:made|taken)', 'substantive', 'Decision made'),
        ]
        
        paragraphs = document_text.split('\n')
        event_id = 0
        
        for para in paragraphs:
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
                
            for pattern, event_type, title in event_patterns:
                import re
                if re.search(pattern, para, re.IGNORECASE):
                    event = MajorEvent(
                        event_id=f"fallback_event_{event_id}",
                        title=title,
                        description=para.strip()[:500],  # Limit description length
                        timestamp="",  # Extract if possible
                        legal_significance=f"Identified {event_type} event",
                        event_type=event_type
                    )
                    events.append(event)
                    event_id += 1
                    break  # Only match first pattern per paragraph
        
        return events
    
    def _filter_significant_events(self, events: List[MajorEvent]) -> List[MajorEvent]:
        """Filter events by legal significance and remove duplicates."""
        if not events:
            return events
        
        # Define significance scores for different event types
        significance_scores = {
            'substantive': 10,   # Judgments, decisions, orders
            'procedural': 5,     # Filings, hearings, notices  
            'evidential': 7,     # Evidence, witness testimony
            'general': 3         # Other events
        }
        
        # Score events
        scored_events = []
        for event in events:
            score = significance_scores.get(event.event_type, 3)
            
            # Boost score for certain keywords in description
            important_keywords = [
                'judgment', 'decision', 'order', 'held', 'ruled', 'concluded',
                'evidence', 'witness', 'testimony', 'precedent', 'law'
            ]
            
            description_lower = event.description.lower()
            for keyword in important_keywords:
                if keyword in description_lower:
                    score += 1
            
            scored_events.append((score, event))
        
        # Sort by score (descending) and take top events
        scored_events.sort(key=lambda x: x[0], reverse=True)
        
        # Remove duplicates based on similar descriptions
        unique_events = []
        seen_descriptions = set()
        
        for score, event in scored_events:
            # Simple deduplication based on first 100 characters
            desc_key = event.description[:100].lower().strip()
            if desc_key not in seen_descriptions:
                unique_events.append(event)
                seen_descriptions.add(desc_key)
        
        # Limit to reasonable number of events
        max_events = 15
        return unique_events[:max_events]
    
    def _create_timeline(self, events: List[MajorEvent]) -> List[str]:
        """Create a chronological timeline of events."""
        timeline = []
        
        # Try to sort events chronologically if timestamps are available
        dated_events = [e for e in events if e.timestamp]
        undated_events = [e for e in events if not e.timestamp]
        
        # For dated events, attempt to sort by timestamp
        # This is simplified - in production, you'd use proper date parsing
        dated_events.sort(key=lambda x: x.timestamp)
        
        # Create timeline entries
        for event in dated_events:
            timeline_entry = f"{event.timestamp}: {event.title}"
            timeline.append(timeline_entry)
        
        # Add undated events at the end
        for event in undated_events:
            timeline_entry = f"[Date not specified]: {event.title}"
            timeline.append(timeline_entry)
        
        return timeline