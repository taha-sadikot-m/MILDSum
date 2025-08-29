"""Preprocessing utilities for legal documents."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import unicodedata

logger = logging.getLogger(__name__)


class LegalDocumentPreprocessor:
    """Utility class for preprocessing legal documents before analysis."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.common_headers = [
            r'page\s+\d+\s+of\s+\d+',
            r'page\s+\d+',
            r'judgment\s+dated',
            r'order\s+dated',
            r'petition\s+no\.?',
            r'case\s+no\.?',
            r'civil\s+appeal\s+no\.?',
            r'criminal\s+appeal\s+no\.?'
        ]
        
        self.section_markers = [
            r'\b(?:facts?|factual\s+background)\b',
            r'\b(?:issues?|legal\s+issues?|questions?)\b',
            r'\b(?:arguments?|submissions?|contentions?)\b',
            r'\b(?:analysis|discussion|reasoning|held)\b',
            r'\b(?:conclusions?|judgment|order|decision)\b',
            r'\b(?:directions?|reliefs?|prayers?)\b'
        ]
    
    def clean_document(self, text: str) -> str:
        """
        Clean and normalize legal document text.
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned and normalized text
        """
        try:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Remove headers and footers
            text = self._remove_headers_footers(text)
            
            # Fix common OCR issues
            text = self._fix_ocr_issues(text)
            
            # Normalize whitespace
            text = self._normalize_whitespace(text)
            
            # Fix punctuation and formatting
            text = self._fix_punctuation(text)
            
            # Remove page numbers and irrelevant content
            text = self._remove_page_numbers(text)
            
            logger.info("Document cleaning completed")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Document cleaning failed: {str(e)}")
            return text  # Return original text if cleaning fails
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip common header patterns
            is_header = any(
                re.search(pattern, line_lower) 
                for pattern in self.common_headers
            )
            
            # Skip very short lines that are likely formatting artifacts
            if len(line.strip()) < 5:
                continue
            
            # Skip lines that are mostly uppercase (often headers)
            if len(line) > 10 and line.isupper():
                continue
            
            if not is_header:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _fix_ocr_issues(self, text: str) -> str:
        """Fix common OCR recognition issues."""
        # Common OCR substitutions
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l to uppercase I
            r'\b0\b': 'O',  # zero to uppercase O in appropriate contexts
            r'rn': 'm',     # rn combination often misread as m
            r'\s+([.,:;!?])': r'\1',  # Remove space before punctuation
            r'([.,:;!?])\s*([a-zA-Z])': r'\1 \2',  # Ensure space after punctuation
            r'(\d)\s*([.]\s*\d)': r'\1\2',  # Fix decimal numbers
            r'([a-z])([A-Z])': r'\1 \2',  # Add space between word boundaries
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple line breaks with double line break
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing spaces from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _fix_punctuation(self, text: str) -> str:
        """Fix punctuation and formatting issues."""
        # Fix common punctuation issues
        punctuation_fixes = [
            (r'\s+([.,:;!?])', r'\1'),  # Remove space before punctuation
            (r'([.,:;!?])([a-zA-Z])', r'\1 \2'),  # Add space after punctuation
            (r'\.{3,}', '...'),  # Normalize multiple dots to ellipsis
            (r'[-−–—]{2,}', ' - '),  # Normalize multiple dashes
            (r'["""]', '"'),  # Normalize quotes
            (r'['']', "'"),  # Normalize apostrophes
        ]
        
        for pattern, replacement in punctuation_fixes:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _remove_page_numbers(self, text: str) -> str:
        """Remove page numbers and similar artifacts."""
        # Remove standalone page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove page X of Y patterns
        text = re.sub(r'\n\s*page\s+\d+\s+of\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove standalone numbers at line endings (likely page numbers)
        text = re.sub(r'\n\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def identify_document_structure(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        Identify the structure of a legal document by finding section boundaries.
        
        Args:
            text: Cleaned document text
            
        Returns:
            Dictionary mapping section names to (start, end) positions
        """
        try:
            sections = {}
            lines = text.split('\n')
            
            # Find section boundaries
            section_boundaries = []
            
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                
                for j, pattern in enumerate(self.section_markers):
                    if re.search(pattern, line_lower):
                        section_name = self._get_section_name(pattern)
                        section_boundaries.append((i, section_name, j))
                        break
            
            # Sort boundaries by line number
            section_boundaries.sort(key=lambda x: x[0])
            
            # Calculate section spans
            for i, (line_num, section_name, pattern_idx) in enumerate(section_boundaries):
                start_line = line_num
                
                # Find end line (start of next section or end of document)
                if i < len(section_boundaries) - 1:
                    end_line = section_boundaries[i + 1][0]
                else:
                    end_line = len(lines)
                
                # Convert to character positions
                start_pos = len('\n'.join(lines[:start_line]))
                end_pos = len('\n'.join(lines[:end_line]))
                
                sections[section_name] = (start_pos, end_pos)
            
            logger.info(f"Identified {len(sections)} document sections")
            return sections
            
        except Exception as e:
            logger.error(f"Document structure identification failed: {str(e)}")
            return {}
    
    def _get_section_name(self, pattern: str) -> str:
        """Map regex pattern to section name."""
        pattern_to_name = {
            r'\b(?:facts?|factual\s+background)\b': 'facts',
            r'\b(?:issues?|legal\s+issues?|questions?)\b': 'legal_issues',
            r'\b(?:arguments?|submissions?|contentions?)\b': 'arguments',
            r'\b(?:analysis|discussion|reasoning|held)\b': 'reasoning',
            r'\b(?:conclusions?|judgment|order|decision)\b': 'conclusion',
            r'\b(?:directions?|reliefs?|prayers?)\b': 'directions'
        }
        
        return pattern_to_name.get(pattern, 'unknown')
    
    def extract_metadata_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata using pattern matching.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with extracted metadata
        """
        try:
            metadata = {
                'case_numbers': [],
                'dates': [],
                'courts': [],
                'parties': [],
                'judges': [],
                'citations': []
            }
            
            # Extract case numbers
            case_patterns = [
                r'(?:case|petition|writ|appeal|civil|criminal)[\s\w]*no\.?\s*([a-z0-9/\-\s]+)',
                r'([a-z]{2,}\s*\d+/\d+)',
                r'(\d+/\d+)'
            ]
            
            for pattern in case_patterns:
                matches = re.findall(pattern, text[:2000], re.IGNORECASE)
                metadata['case_numbers'].extend([m.strip() for m in matches])
            
            # Extract dates
            date_patterns = [
                r'(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})',
                r'(\d{1,2}\w{0,2}\s+\w+\s+\d{4})',
                r'(\w+\s+\d{1,2},?\s+\d{4})'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text[:2000])
                metadata['dates'].extend(matches)
            
            # Extract court names
            court_patterns = [
                r'(?:in the|before the)\s+([^\\n]+(?:court|tribunal)[^\\n]*)',
                r'([^\\n]*(?:high court|supreme court|district court)[^\\n]*)'
            ]
            
            for pattern in court_patterns:
                matches = re.findall(pattern, text[:1000], re.IGNORECASE)
                metadata['courts'].extend([m.strip() for m in matches])
            
            # Extract party names (simple pattern)
            party_patterns = [
                r'([a-z][a-z\s]+)\s+vs?\.\s+([a-z][a-z\s]+)',
                r'([a-z][a-z\s]+)\s+v\.\s+([a-z][a-z\s]+)'
            ]
            
            for pattern in party_patterns:
                matches = re.findall(pattern, text[:1000], re.IGNORECASE)
                for match in matches:
                    metadata['parties'].extend([p.strip() for p in match])
            
            # Extract citations
            citation_patterns = [
                r'(\d{4}\s+[a-z]+\s+\d+)',
                r'(air\s+\d{4}\s+[a-z]+\s+\d+)',
                r'([a-z]+\s+\d{4}\s+[a-z]+\s+\d+)'
            ]
            
            for pattern in citation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                metadata['citations'].extend(matches)
            
            # Clean and deduplicate
            for key in metadata:
                metadata[key] = list(set([item for item in metadata[key] if item and len(item.strip()) > 2]))
            
            logger.info("Metadata extraction completed")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return {}
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling legal document specifics.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            # Define sentence ending patterns for legal text
            sentence_endings = r'[.!?]+(?=\s+[A-Z]|\s*$)'
            
            # Split into potential sentences
            sentences = re.split(sentence_endings, text)
            
            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip very short sentences (likely formatting artifacts)
                if len(sentence) < 10:
                    continue
                
                # Skip sentences that are mostly numbers or punctuation
                word_count = len(re.findall(r'\b\w+\b', sentence))
                if word_count < 3:
                    continue
                
                cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.error(f"Sentence splitting failed: {str(e)}")
            return [text]  # Return original text as single sentence if splitting fails
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities like acts, sections, case names.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their instances
        """
        try:
            entities = {
                'acts': [],
                'sections': [],
                'articles': [],
                'rules': [],
                'case_names': [],
                'legal_provisions': []
            }
            
            # Extract Acts
            act_patterns = [
                r'\b([a-z][a-z\s,]+act[\s,]*\d{4})\b',
                r'\b([a-z][a-z\s]+act)\b'
            ]
            
            for pattern in act_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['acts'].extend([m.strip() for m in matches if len(m.strip()) > 5])
            
            # Extract Sections
            section_patterns = [
                r'\bsection\s+(\d+[a-z]*(?:\([^)]+\))?)',
                r'\bsec\.?\s+(\d+[a-z]*)',
                r'\bs\.?\s+(\d+[a-z]*)'
            ]
            
            for pattern in section_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['sections'].extend(matches)
            
            # Extract Articles
            article_patterns = [
                r'\barticle\s+(\d+[a-z]*(?:\([^)]+\))?)',
                r'\bart\.?\s+(\d+[a-z]*)'
            ]
            
            for pattern in article_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['articles'].extend(matches)
            
            # Extract Rules
            rule_patterns = [
                r'\brule\s+(\d+[a-z]*(?:\([^)]+\))?)',
                r'\br\.?\s+(\d+[a-z]*)'
            ]
            
            for pattern in rule_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['rules'].extend(matches)
            
            # Extract case names
            case_name_patterns = [
                r'\b([a-z][a-z\s]+\s+v\.?\s+[a-z][a-z\s]+)\b',
                r'\b([a-z]+\s+vs?\.\s+[a-z]+)\b'
            ]
            
            for pattern in case_name_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities['case_names'].extend([m.strip() for m in matches])
            
            # Clean and deduplicate
            for key in entities:
                entities[key] = list(set([
                    item.strip() for item in entities[key] 
                    if item and len(item.strip()) > 2
                ]))
            
            logger.info("Legal entity extraction completed")
            return entities
            
        except Exception as e:
            logger.error(f"Legal entity extraction failed: {str(e)}")
            return {}


def preprocess_batch(documents: List[str]) -> List[str]:
    """
    Preprocess a batch of legal documents.
    
    Args:
        documents: List of document texts
        
    Returns:
        List of preprocessed document texts
    """
    preprocessor = LegalDocumentPreprocessor()
    return [preprocessor.clean_document(doc) for doc in documents]


def extract_key_phrases(text: str, max_phrases: int = 20) -> List[str]:
    """
    Extract key phrases from legal text using simple pattern matching.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to return
        
    Returns:
        List of key phrases
    """
    try:
        # Find noun phrases and legal terms
        phrase_patterns = [
            r'\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b',  # Capitalized phrases
            r'\b[a-z]+\s+(?:act|law|rule|section|article|provision)\b',  # Legal terms
            r'\b(?:supreme court|high court|district court)\b',  # Courts
            r'\b[a-z]+\s+v\.?\s+[a-z]+\b'  # Case names
        ]
        
        phrases = []
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend([m.strip() for m in matches])
        
        # Filter and rank phrases
        phrase_counts = {}
        for phrase in phrases:
            if len(phrase) > 3 and len(phrase.split()) <= 5:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Sort by frequency and return top phrases
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in sorted_phrases[:max_phrases]]
        
    except Exception as e:
        logger.error(f"Key phrase extraction failed: {str(e)}")
        return []