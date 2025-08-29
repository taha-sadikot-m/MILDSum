"""Prompt templates for legal document processing."""

from typing import Dict, Any


class LegalPrompts:
    """Collection of prompt templates for legal document analysis."""
    
    # Document Preprocessor prompts
    DOCUMENT_STRUCTURE_ANALYSIS = """
Analyze the following Indian court judgment and extract its structural components.
Identify the key sections and extract metadata.

Document:
{document_text}

Please provide a structured analysis in JSON format with:
1. case_metadata: case number, court, date, parties involved
2. document_sections: facts, legal_issues, reasoning, conclusion
3. text_quality: readability score, complexity assessment
4. preprocessing_notes: any cleaning requirements or structural issues

Focus on legal accuracy and preserve important legal terminology.
"""

    SECTION_IDENTIFICATION = """
Given this legal judgment text, identify and extract the main sections:

{document_text}

Return a JSON object with clearly separated sections:
- facts: The factual background of the case
- legal_issues: The legal questions to be resolved  
- reasoning: The court's legal analysis and reasoning
- conclusion: The final judgment/order

Ensure each section maintains the original legal language and terminology.
"""

    # Event Extraction prompts  
    MAJOR_EVENT_EXTRACTION = """
Extract the major legal events from this court judgment in chronological order.
Focus on significant procedural and substantive developments.

Judgment text:
{document_text}

Provide a JSON array of major events with:
- event_id: unique identifier
- title: brief descriptive title
- description: detailed description of the event
- timestamp: when the event occurred (if mentioned)
- legal_significance: why this event is legally important
- event_type: (procedural, substantive, evidential, etc.)

Focus on events that materially impact the case outcome.
"""

    SUB_EVENT_ANALYSIS = """
For the major event described below, extract detailed sub-events and supporting information:

Major Event: {major_event}

Context from judgment: {context}

Provide JSON with:
- sub_events: array of detailed sub-events within this major event
- evidence: supporting evidence or documentation mentioned
- legal_references: citations, precedents, or legal provisions referenced
- causal_relationships: how this event relates to other events
- participants: parties, officials, or entities involved

Maintain legal precision and cite specific details from the text.
"""

    # Knowledge Graph prompts
    KNOWLEDGE_GRAPH_CONSTRUCTION = """
Create a structured knowledge representation of this legal case:

Events: {events_data}
Case Context: {case_context}

Generate a JSON knowledge graph with:
- entities: all relevant legal entities (parties, courts, laws, precedents)
- relationships: connections between entities and events
- temporal_sequence: chronological ordering of events
- legal_hierarchy: relationships between legal concepts
- citation_network: connections to referenced cases/laws

Ensure the graph captures the legal reasoning flow and dependencies.
"""

    # Summary Generation prompts
    SUMMARY_GENERATION = """
Generate a comprehensive legal summary based on the extracted events and analysis:

Case Metadata: {metadata}
Major Events: {major_events}  
Legal Analysis: {legal_reasoning}

Create a coherent abstractive summary that:
1. Captures all key legal points
2. Maintains chronological flow
3. Explains the legal reasoning
4. States the final outcome clearly
5. Uses appropriate legal terminology

The summary should be comprehensive yet concise, suitable for legal professionals.
Length: approximately {target_length} words.
"""

    SUMMARY_REFINEMENT = """
Refine this legal summary to improve clarity and completeness:

Original Summary: {original_summary}
Key Events: {key_events}
Legal Issues: {legal_issues}

Enhance the summary by:
1. Ensuring all major legal points are covered
2. Improving logical flow and coherence
3. Adding missing crucial details
4. Maintaining legal accuracy
5. Optimizing for readability while preserving precision

Return the refined summary maintaining the same approximate length.
"""

    # Quality Evaluation prompts
    QUALITY_EVALUATION = """
Evaluate the quality of this legal summary:

Summary: {summary}
Original Judgment: {original_text}
Key Events: {key_events}

Assess the summary on:
1. Legal Accuracy: correctness of legal facts and reasoning
2. Completeness: coverage of all major points
3. Coherence: logical flow and structure
4. Clarity: readability and understandability
5. Legal Language: appropriate use of legal terminology

Provide scores (1-10) for each dimension and specific improvement suggestions.
Return evaluation in JSON format.
"""

    SUMMARY_COMPARISON = """
Compare these two summaries and determine which better represents the original judgment:

Summary A: {summary_a}
Summary B: {summary_b}
Original Judgment Key Points: {key_points}

Evaluate based on:
- Factual accuracy
- Legal completeness  
- Clarity and coherence
- Appropriate legal language

Provide detailed comparison and recommendation in JSON format.
"""

    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """Format a prompt template with provided variables."""
        return template.format(**kwargs)
    
    @staticmethod
    def get_system_instruction(agent_type: str) -> str:
        """Get system instruction for specific agent type."""
        instructions = {
            "preprocessor": """You are a legal document analysis expert specializing in Indian court judgments. 
                            Your role is to analyze document structure, extract metadata, and identify key sections 
                            with high precision. Maintain legal terminology and accuracy.""",
            
            "event_extractor": """You are a legal events specialist focused on extracting chronological sequences 
                               of major legal developments from court judgments. Identify procedural and substantive 
                               events that are legally significant.""",
            
            "sub_event_analyzer": """You are a detailed legal analysis expert who extracts fine-grained sub-events, 
                                  evidence, and legal references. Focus on causal relationships and supporting details 
                                  that provide legal context.""",
            
            "knowledge_builder": """You are a legal knowledge graph architect who structures legal information into 
                                 hierarchical relationships. Create comprehensive representations of legal entities, 
                                 relationships, and reasoning flows.""",
            
            "summary_generator": """You are a legal writing expert specializing in creating high-quality abstractive 
                                 summaries of court judgments. Generate comprehensive yet concise summaries that 
                                 capture essential legal reasoning and outcomes.""",
            
            "quality_evaluator": """You are a legal quality assurance expert who evaluates summary accuracy, 
                                 completeness, and adherence to legal standards. Provide detailed assessments 
                                 and improvement recommendations."""
        }
        
        return instructions.get(agent_type, "You are a legal AI assistant focused on accurate legal document analysis.")