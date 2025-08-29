"""Document Preprocessor Agent for legal document analysis."""

import logging
import re
from typing import Dict, Any, List, Optional
import asyncio

from ..models.gemini_client import GeminiClient
from ..models.prompts import LegalPrompts
from ..workflow.states import LegalSummarizationState, CaseMetadata, DocumentSections
from ..config.settings import settings

logger = logging.getLogger(__name__)


class DocumentPreprocessorAgent:
    """
    Agent responsible for cleaning and structuring input court judgment text.
    Identifies key sections and extracts metadata.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """Initialize the Document Preprocessor Agent."""
        self.gemini_client = gemini_client or GeminiClient()
        self.agent_type = "preprocessor"
        
    async def process(self, state: LegalSummarizationState) -> LegalSummarizationState:
        """
        Process the legal document to extract structure and metadata.
        
        Args:
            state: Current workflow state containing the document text
            
        Returns:
            Updated state with extracted metadata and sections
        """
        try:
            logger.info(f"Starting document preprocessing for document: {state.document_id}")
            state.update_stage("preprocessing", self.agent_type)
            
            # Clean the document text
            cleaned_text = self._clean_document_text(state.document_text)
            
            # Extract metadata and structure in parallel
            metadata_task = self._extract_metadata(cleaned_text)
            sections_task = self._extract_sections(cleaned_text)
            
            metadata, sections = await asyncio.gather(metadata_task, sections_task)
            
            # Update state with results
            state.case_metadata = metadata
            state.document_sections = sections
            state.document_text = cleaned_text  # Use cleaned version
            
            # Add preprocessing notes
            state.preprocessing_notes = self._generate_preprocessing_notes(
                state.document_text, metadata, sections
            )
            
            logger.info("Document preprocessing completed successfully")
            return state
            
        except Exception as e:
            error_msg = f"Error in document preprocessing: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    def _clean_document_text(self, text: str) -> str:
        """Clean and normalize the document text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\d+)\.(\d+)', r'\1. \2', text)  # Fix numbering
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    async def _extract_metadata(self, document_text: str) -> CaseMetadata:
        """Extract case metadata from the document."""
        prompt = LegalPrompts.format_prompt(
            LegalPrompts.DOCUMENT_STRUCTURE_ANALYSIS,
            document_text=document_text[:3000]  # Use first 3000 chars for metadata
        )
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        try:
            response = await self.gemini_client.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                schema={
                    "case_metadata": {
                        "case_number": "string",
                        "court": "string", 
                        "date": "string",
                        "parties": ["string"],
                        "judges": ["string"],
                        "case_type": "string"
                    }
                }
            )
            
            metadata_dict = response.get("case_metadata", {})
            return CaseMetadata(**metadata_dict)
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata with LLM: {str(e)}, using fallback")
            return self._extract_metadata_fallback(document_text)
    
    def _extract_metadata_fallback(self, document_text: str) -> CaseMetadata:
        """Fallback method to extract metadata using regex patterns."""
        metadata = CaseMetadata()
        
        # Extract case number (common patterns)
        case_number_patterns = [
            r'(?:Case|Petition|Writ|Appeal|Civil|Criminal)[\s\w]*No\.?\s*([A-Z0-9/\-\s]+)',
            r'([A-Z]{2,}\s*\d+/\d+)',
            r'(\d+/\d+)'
        ]
        
        for pattern in case_number_patterns:
            match = re.search(pattern, document_text[:1000], re.IGNORECASE)
            if match:
                metadata.case_number = match.group(1).strip()
                break
        
        # Extract court name
        court_patterns = [
            r'(?:IN THE|BEFORE THE)\s+([^\\n]+(?:COURT|TRIBUNAL)[^\\n]*)',
            r'([^\\n]*(?:HIGH COURT|SUPREME COURT|DISTRICT COURT)[^\\n]*)'
        ]
        
        for pattern in court_patterns:
            match = re.search(pattern, document_text[:500], re.IGNORECASE)
            if match:
                metadata.court = match.group(1).strip()
                break
        
        # Extract date
        date_patterns = [
            r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})',
            r'(\d{1,2}\w{0,2}\s+\w+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, document_text[:1000])
            if match:
                metadata.date = match.group(1).strip()
                break
        
        return metadata
    
    async def _extract_sections(self, document_text: str) -> DocumentSections:
        """Extract and identify main sections of the document."""
        prompt = LegalPrompts.format_prompt(
            LegalPrompts.SECTION_IDENTIFICATION,
            document_text=document_text
        )
        
        system_instruction = LegalPrompts.get_system_instruction(self.agent_type)
        
        try:
            response = await self.gemini_client.generate_structured_output(
                prompt=prompt,
                system_instruction=system_instruction,
                schema={
                    "facts": "string",
                    "legal_issues": "string",
                    "reasoning": "string", 
                    "conclusion": "string",
                    "procedural_history": "string"
                }
            )
            
            return DocumentSections(**response)
            
        except Exception as e:
            logger.warning(f"Failed to extract sections with LLM: {str(e)}, using fallback")
            return self._extract_sections_fallback(document_text)
    
    def _extract_sections_fallback(self, document_text: str) -> DocumentSections:
        """Fallback method to extract sections using pattern matching."""
        sections = DocumentSections()
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in document_text.split('\n') if p.strip()]
        
        # Simple heuristic-based section detection
        current_section = "facts"
        facts_paras = []
        issues_paras = []
        reasoning_paras = []
        conclusion_paras = []
        
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            
            # Detect section transitions
            if any(keyword in para_lower for keyword in ['issue', 'question', 'law', 'legal']):
                current_section = "legal_issues"
            elif any(keyword in para_lower for keyword in ['reasoning', 'analysis', 'discussion', 'held']):
                current_section = "reasoning"
            elif any(keyword in para_lower for keyword in ['conclusion', 'judgment', 'order', 'disposed']):
                current_section = "conclusion"
            
            # Assign paragraph to current section
            if current_section == "facts":
                facts_paras.append(para)
            elif current_section == "legal_issues":
                issues_paras.append(para)
            elif current_section == "reasoning":
                reasoning_paras.append(para)
            elif current_section == "conclusion":
                conclusion_paras.append(para)
        
        sections.facts = ' '.join(facts_paras)
        sections.legal_issues = ' '.join(issues_paras)
        sections.reasoning = ' '.join(reasoning_paras)
        sections.conclusion = ' '.join(conclusion_paras)
        
        return sections
    
    def _generate_preprocessing_notes(
        self, 
        cleaned_text: str, 
        metadata: CaseMetadata, 
        sections: DocumentSections
    ) -> List[str]:
        """Generate notes about the preprocessing results."""
        notes = []
        
        # Document length analysis
        word_count = len(cleaned_text.split())
        notes.append(f"Document contains {word_count} words")
        
        # Metadata completeness
        if not metadata.case_number:
            notes.append("Case number not found - may affect citation")
        
        if not metadata.court:
            notes.append("Court name not identified clearly")
        
        # Section analysis
        section_lengths = {
            "facts": len(sections.facts.split()),
            "legal_issues": len(sections.legal_issues.split()),
            "reasoning": len(sections.reasoning.split()),
            "conclusion": len(sections.conclusion.split())
        }
        
        for section, length in section_lengths.items():
            if length == 0:
                notes.append(f"No content identified for {section} section")
            elif length < 50:
                notes.append(f"{section} section appears to be very short ({length} words)")
        
        return notes