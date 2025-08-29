"""Knowledge Graph Builder Agent for legal case analysis."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import asyncio

from ..models.gemini_client import GeminiClient
from ..models.prompts import LegalPrompts
from ..workflow.states import LegalSummarizationState, MajorEvent, LegalReasoning
from ..config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeBuilderAgent:
    """
    Agent responsible for structuring events in hierarchical JSON format.
    Creates relationships between events and legal concepts.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the Knowledge Builder Agent."""
        self.gemini_client = gemini_client or GeminiClient()
        self.agent_type = "knowledge_builder"
        
    async def process(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """
        Build knowledge graph from extracted events and document analysis.
        
        Args:
            state: Current workflow state with events and sections
            
        Returns:
            Updated state with knowledge graph and legal reasoning
        """
        try:
            logger.info(f"Building knowledge graph for {len(state.major_events)} events")
            state.update_stage("knowledge_graph_construction", self.agent_type)
            
            # Extract legal reasoning components
            legal_reasoning = await self._extract_legal_reasoning(state)
            
            # Build the knowledge graph
            knowledge_graph = await self._build_knowledge_graph(state)
            
            # Update state
            state.legal_reasoning = legal_reasoning
            state.knowledge_graph = knowledge_graph
            
            logger.info("Knowledge graph construction completed")
            return state
            
        except Exception as e:
            error_msg = f"Error in knowledge graph construction: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def _extract_legal_reasoning(self, state: LegalSummarizationState) -> LegalReasoning:
        """Extract legal reasoning components from the case."""
        try:
            # Prepare context from document sections and events
            reasoning_context = self._prepare_reasoning_context(state)
            
            prompt = f"""
Analyze the following legal case information and extract the key legal reasoning components:

Case Context:
{reasoning_context}

Extract and structure:
1. Key legal issues that need to be resolved
2. Legal precedents and cases cited
3. Fundamental legal principles applied
4. Logical flow of the court's reasoning

Provide a structured JSON response with these components.
"""
            
            system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
            
            response = await self.gemini_client.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                schema={
                    "key_issues": ["string"],
                    "precedents_cited": ["string"],
                    "legal_principles": ["string"],
                    "reasoning_flow": ["string"]
                }
            )
            
            return LegalReasoning(
                key_issues=response.get("key_issues", []),
                precedents_cited=response.get("precedents_cited", []),
                legal_principles=response.get("legal_principles", []),
                reasoning_flow=response.get("reasoning_flow", [])
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract legal reasoning with LLM: {str(e)}, using fallback")
            return self._extract_legal_reasoning_fallback(state)
    
    def _prepare_reasoning_context(self, state: LegalSummarizationState) -> str:
        """Prepare context for legal reasoning extraction."""
        context_parts = []
        
        # Add document sections
        if state.document_sections.legal_issues:
            context_parts.append(f"Legal Issues:\n{state.document_sections.legal_issues}\n")
        
        if state.document_sections.reasoning:
            context_parts.append(f"Court's Reasoning:\n{state.document_sections.reasoning}\n")
        
        # Add key events
        if state.major_events:
            events_summary = []
            for event in state.major_events[:5]:  # Top 5 events
                events_summary.append(f"- {event.title}: {event.description[:200]}")
            context_parts.append(f"Key Events:\n" + "\n".join(events_summary) + "\n")
        
        return "\n".join(context_parts)
    
    def _extract_legal_reasoning_fallback(self, state: LegalSummarizationState) -> LegalReasoning:
        """Fallback method to extract legal reasoning using pattern matching."""
        import re
        
        # Combine relevant text sources
        text_sources = [
            state.document_sections.legal_issues,
            state.document_sections.reasoning,
            state.document_text[:5000]  # First 5000 chars
        ]
        
        combined_text = " ".join(filter(None, text_sources))
        
        # Extract legal issues
        issue_patterns = [
            r'(?:issue|question|matter)(?:\s+is|\s+whether|\s+of)([^.!?]+)',
            r'(?:whether|if)([^.!?]+)',
            r'(?:determination|deciding|resolution)(?:\s+of|\s+on)([^.!?]+)'
        ]
        
        key_issues = []
        for pattern in issue_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            key_issues.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        # Extract precedents
        precedent_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'((?:AIR|SCC|SCR)\s+\d{4}\s+[A-Z]+\s+\d+)',
            r'(\d{4}\s+[A-Z]+\s+\d+)'
        ]
        
        precedents_cited = []
        for pattern in precedent_patterns:
            matches = re.findall(pattern, combined_text)
            precedents_cited.extend(matches)
        
        # Extract legal principles
        principle_patterns = [
            r'(?:principle|rule|doctrine|law)(?:\s+is|\s+states|\s+provides)([^.!?]+)',
            r'(?:established|settled|well-known)(?:\s+principle|\s+law|\s+rule)([^.!?]+)',
            r'(?:section|article)\s+\d+[^.!?]*'
        ]
        
        legal_principles = []
        for pattern in principle_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            legal_principles.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return LegalReasoning(
            key_issues=key_issues[:10],
            precedents_cited=list(set(precedents_cited))[:15],
            legal_principles=legal_principles[:10],
            reasoning_flow=[]
        )
    
    async def _build_knowledge_graph(self, state: LegalSummarizationState) -> Dict[str, Any]:
        """Build comprehensive knowledge graph from case data."""
        try:
            # Prepare events data for graph construction
            events_data = self._serialize_events_for_graph(state.major_events)
            case_context = self._prepare_case_context(state)
            
            prompt = LegalPrompts.format_prompt(
                LegalPrompts.KNOWLEDGE_GRAPH_CONSTRUCTION,
                events_data=events_data,
                case_context=case_context
            )
            
            system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
            
            response = await self.gemini_client.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                schema={
                    "entities": {
                        "parties": ["string"],
                        "courts": ["string"],
                        "laws": ["string"],
                        "precedents": ["string"],
                        "legal_concepts": ["string"]
                    },
                    "relationships": [
                        {
                            "source": "string",
                            "target": "string",
                            "relationship_type": "string",
                            "description": "string"
                        }
                    ],
                    "temporal_sequence": [
                        {
                            "event_id": "string",
                            "sequence_order": "number",
                            "dependencies": ["string"]
                        }
                    ],
                    "legal_hierarchy": {
                        "constitutional_provisions": ["string"],
                        "statutory_provisions": ["string"],
                        "case_law": ["string"],
                        "regulations": ["string"]
                    }
                }
            )
            
            # Enhance with computed relationships
            enhanced_graph = self._enhance_knowledge_graph(response, state)
            
            return enhanced_graph
            
        except Exception as e:
            logger.warning(f"Failed to build knowledge graph with LLM: {str(e)}, using fallback")
            return self._build_knowledge_graph_fallback(state)
    
    def _serialize_events_for_graph(self, events: List[MajorEvent]) -> str:
        """Serialize events data for knowledge graph construction."""
        events_summary = []
        
        for event in events:
            event_summary = {
                "id": event.event_id,
                "title": event.title,
                "type": event.event_type,
                "description": event.description[:300],  # Truncate for context
                "sub_events_count": len(event.sub_events),
                "legal_references": []
            }
            
            # Extract legal references from sub-events
            for sub_event in event.sub_events:
                event_summary["legal_references"].extend(sub_event.legal_references)
            
            events_summary.append(event_summary)
        
        import json
        return json.dumps(events_summary, indent=2)
    
    def _prepare_case_context(self, state: LegalSummarizationState) -> str:
        """Prepare case context for knowledge graph construction."""
        context_parts = []
        
        # Case metadata
        if state.case_metadata:
            metadata_str = f"""
Case: {state.case_metadata.case_number}
Court: {state.case_metadata.court}
Parties: {', '.join(state.case_metadata.parties)}
"""
            context_parts.append(metadata_str)
        
        # Key sections
        if state.document_sections.facts:
            context_parts.append(f"Facts: {state.document_sections.facts[:500]}")
        
        if state.document_sections.conclusion:
            context_parts.append(f"Conclusion: {state.document_sections.conclusion[:500]}")
        
        return "\n\n".join(context_parts)
    
    def _enhance_knowledge_graph(
        self, 
        base_graph: Dict[str, Any], 
        state: LegalSummarizationState
    ) -> Dict[str, Any]:
        """Enhance the knowledge graph with computed relationships and metrics."""
        enhanced_graph = base_graph.copy()
        
        # Add event connectivity analysis
        enhanced_graph["event_connectivity"] = self._analyze_event_connectivity(state.major_events)
        
        # Add legal concept frequency
        enhanced_graph["concept_frequency"] = self._analyze_concept_frequency(state)
        
        # Add citation network
        enhanced_graph["citation_network"] = self._build_citation_network(state)
        
        # Add temporal dependencies
        enhanced_graph["temporal_dependencies"] = self._analyze_temporal_dependencies(state.major_events)
        
        return enhanced_graph
    
    def _analyze_event_connectivity(self, events: List[MajorEvent]) -> Dict[str, Any]:
        """Analyze connectivity between events."""
        connectivity = {
            "event_relationships": [],
            "clusters": [],
            "central_events": []
        }
        
        # Simple connectivity analysis based on shared elements
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                # Check for shared legal references, participants, etc.
                shared_refs = self._find_shared_references(event1, event2)
                
                if shared_refs:
                    connectivity["event_relationships"].append({
                        "event1_id": event1.event_id,
                        "event2_id": event2.event_id,
                        "shared_elements": shared_refs,
                        "connection_strength": len(shared_refs)
                    })
        
        return connectivity
    
    def _find_shared_references(self, event1: MajorEvent, event2: MajorEvent) -> List[str]:
        """Find shared references between two events."""
        shared = []
        
        # Collect all references from both events
        refs1 = set()
        refs2 = set()
        
        for sub_event in event1.sub_events:
            refs1.update(sub_event.legal_references)
            refs1.update(sub_event.participants)
        
        for sub_event in event2.sub_events:
            refs2.update(sub_event.legal_references)
            refs2.update(sub_event.participants)
        
        # Find intersection
        shared = list(refs1.intersection(refs2))
        
        return shared
    
    def _analyze_concept_frequency(self, state: LegalSummarizationState) -> Dict[str, int]:
        """Analyze frequency of legal concepts in the case."""
        import re
        from collections import Counter
        
        # Combine all text sources
        all_text = " ".join([
            state.document_text,
            " ".join([event.description for event in state.major_events])
        ])
        
        # Define legal concept patterns
        concept_patterns = [
            (r'\b(?:contract|agreement|covenant)\b', 'contract_law'),
            (r'\b(?:tort|negligence|liability)\b', 'tort_law'),
            (r'\b(?:constitutional|fundamental rights?)\b', 'constitutional_law'),
            (r'\b(?:criminal|penal|prosecution)\b', 'criminal_law'),
            (r'\b(?:property|ownership|title)\b', 'property_law'),
            (r'\b(?:procedure|process|jurisdiction)\b', 'procedural_law'),
            (r'\b(?:evidence|witness|testimony)\b', 'evidence_law'),
            (r'\b(?:appeal|review|revision)\b', 'appellate_law')
        ]
        
        concept_counts = Counter()
        
        for pattern, concept in concept_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            concept_counts[concept] = len(matches)
        
        return dict(concept_counts)
    
    def _build_citation_network(self, state: LegalSummarizationState) -> Dict[str, Any]:
        """Build citation network from legal references."""
        citations = []
        
        # Collect all legal references
        for event in state.major_events:
            for sub_event in event.sub_events:
                for ref in sub_event.legal_references:
                    citations.append({
                        "citation": ref,
                        "source_event": event.event_id,
                        "context": sub_event.description[:100]
                    })
        
        # Add precedents from legal reasoning
        if state.legal_reasoning:
            for precedent in state.legal_reasoning.precedents_cited:
                citations.append({
                    "citation": precedent,
                    "source_event": "legal_reasoning",
                    "context": "Court precedent"
                })
        
        return {
            "citations": citations,
            "citation_count": len(citations),
            "unique_citations": len(set(c["citation"] for c in citations))
        }
    
    def _analyze_temporal_dependencies(self, events: List[MajorEvent]) -> List[Dict[str, Any]]:
        """Analyze temporal dependencies between events."""
        dependencies = []
        
        # Simple temporal analysis based on event types and descriptions
        procedural_events = [e for e in events if e.event_type == "procedural"]
        substantive_events = [e for e in events if e.event_type == "substantive"]
        
        # Procedural events typically depend on earlier procedural events
        for i, event in enumerate(procedural_events[1:], 1):
            dependencies.append({
                "dependent_event": event.event_id,
                "prerequisite_event": procedural_events[i-1].event_id,
                "dependency_type": "procedural_sequence",
                "strength": 0.8
            })
        
        # Substantive events typically depend on procedural events
        for sub_event in substantive_events:
            for proc_event in procedural_events:
                if any(word in sub_event.description.lower() 
                       for word in proc_event.title.lower().split()):
                    dependencies.append({
                        "dependent_event": sub_event.event_id,
                        "prerequisite_event": proc_event.event_id,
                        "dependency_type": "procedural_to_substantive",
                        "strength": 0.6
                    })
        
        return dependencies
    
    def _build_knowledge_graph_fallback(self, state: LegalSummarizationState) -> Dict[str, Any]:
        """Fallback method to build basic knowledge graph."""
        # Extract basic entities and relationships
        entities = self._extract_basic_entities(state)
        relationships = self._extract_basic_relationships(state)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "temporal_sequence": [
                {"event_id": event.event_id, "sequence_order": i, "dependencies": []}
                for i, event in enumerate(state.major_events)
            ],
            "legal_hierarchy": {
                "constitutional_provisions": [],
                "statutory_provisions": [],
                "case_law": [],
                "regulations": []
            },
            "event_connectivity": {"event_relationships": [], "clusters": [], "central_events": []},
            "concept_frequency": {},
            "citation_network": {"citations": [], "citation_count": 0, "unique_citations": 0}
        }
    
    def _extract_basic_entities(self, state: LegalSummarizationState) -> Dict[str, List[str]]:
        """Extract basic entities using simple pattern matching."""
        import re
        
        text = state.document_text
        
        # Extract parties (capitalized names)
        parties = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:vs?\.?|v\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        
        # Extract courts
        courts = re.findall(r'\b(?:Supreme Court|High Court|District Court|Tribunal)[^.]*', text, re.IGNORECASE)
        
        return {
            "parties": list(set(parties[:10])),
            "courts": list(set(courts[:5])),
            "laws": [],
            "precedents": [],
            "legal_concepts": []
        }
    
    def _extract_basic_relationships(self, state: LegalSummarizationState) -> List[Dict[str, str]]:
        """Extract basic relationships between entities."""
        relationships = []
        
        # Basic relationship between events
        for i, event in enumerate(state.major_events[:-1]):
            relationships.append({
                "source": event.event_id,
                "target": state.major_events[i+1].event_id,
                "relationship_type": "temporal_sequence",
                "description": "Follows in chronological order"
            })
        
        return relationships