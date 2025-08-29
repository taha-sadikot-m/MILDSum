# Legal Summarization (L-SUMM) Agentic Workflow

An advanced multi-agent system built with LangGraph and LangChain for generating high-quality abstractive summaries of Indian court case judgments using Google's Gemini LLM.

## 🎯 Overview

This project implements a comprehensive agentic workflow that processes legal documents through six specialized agents, each focusing on a specific aspect of legal document analysis and summarization. The system is optimized for ROUGE-2, ROUGE-L, and BERTScore metrics while maintaining legal accuracy and completeness.

## 🏗️ Architecture

### Multi-Agent System

The workflow consists of six specialized agents:

1. **Document Preprocessor Agent** - Cleans and structures input court judgment text, identifies key sections, and extracts metadata
2. **Major Event Extractor Agent** - Identifies primary legal events and proceedings, extracts chronological sequences
3. **Sub-Event Analyzer Agent** - Extracts detailed sub-events for each major event, identifies causal relationships
4. **Knowledge Graph Builder Agent** - Structures events in hierarchical JSON format, creates relationships between events and legal concepts
5. **Summary Generator Agent** - Uses extracted events to generate coherent abstractive summaries optimized for legal domain
6. **Quality Evaluator Agent** - Self-evaluates summary quality, checks for legal accuracy and completeness, refines summary if needed

### Workflow Flow

```
Document Input → Preprocessor → Event Extractor → Sub-Event Analyzer → Knowledge Builder → Summary Generator → Quality Evaluator → Final Summary
```

## 📁 Project Structure

```
legal_summarization/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── preprocessor.py           # Document preprocessing agent
│   ├── event_extractor.py        # Major event extraction
│   ├── sub_event_analyzer.py     # Sub-event analysis
│   ├── knowledge_builder.py      # Knowledge graph construction
│   ├── summary_generator.py      # Summary generation
│   └── quality_evaluator.py      # Quality assessment
├── workflow/
│   ├── __init__.py
│   ├── graph.py                  # LangGraph workflow orchestration
│   └── states.py                 # State management
├── models/
│   ├── __init__.py
│   ├── gemini_client.py          # Gemini LLM integration
│   └── prompts.py                # Prompt templates
├── utils/
│   ├── __init__.py
│   ├── json_storage.py           # JSON storage utilities
│   ├── evaluation.py             # Evaluation metrics
│   └── preprocessing.py          # Text preprocessing utilities
├── config/
│   ├── __init__.py
│   └── settings.py               # Configuration management
├── examples/
│   ├── sample_judgment.txt       # Sample legal document
│   └── run_workflow.py           # Usage example
├── tests/
│   ├── __init__.py
│   └── test_workflow.py          # Comprehensive tests
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Google API key for Gemini access

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/legal-summarization.git
cd legal-summarization
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

4. **Create a `.env` file (optional):**
```env
GOOGLE_API_KEY=your-gemini-api-key
```

## 📖 Usage

### Basic Usage

```python
import asyncio
from legal_summarization import LegalSummarizationWorkflow

async def main():
    # Initialize workflow
    workflow = LegalSummarizationWorkflow()
    
    # Process a legal document
    with open("path/to/judgment.txt", "r") as f:
        document_text = f.read()
    
    result = await workflow.process_document(
        document_text=document_text,
        document_id="case_123",
        target_length=400
    )
    
    # Access results
    print(f"Generated Summary: {result.generated_summary}")
    print(f"Quality Score: {result.evaluation_scores.legal_metrics.overall_score}")
    print(f"Events Extracted: {len(result.major_events)}")

# Run the workflow
asyncio.run(main())
```

### Synchronous Usage

```python
from legal_summarization import LegalSummarizationWorkflow

# Initialize workflow
workflow = LegalSummarizationWorkflow()

# Process document synchronously
result = workflow.process_document_sync(
    document_text="Your legal document text here...",
    document_id="case_456",
    target_length=300
)

print(result.generated_summary)
```

### Batch Processing

```python
import asyncio
from legal_summarization import LegalSummarizationWorkflow

async def process_batch():
    workflow = LegalSummarizationWorkflow()
    
    documents = [
        {"text": "Document 1 text...", "id": "doc_1"},
        {"text": "Document 2 text...", "id": "doc_2"},
    ]
    
    results = await workflow.process_batch(documents, batch_size=2)
    
    for result in results:
        print(f"Document {result.document_id}: {result.generated_summary}")

asyncio.run(process_batch())
```

### Using Storage and Evaluation

```python
from legal_summarization.utils import JSONStorageManager, LegalSummarizationEvaluator

# Save results
storage = JSONStorageManager()
result_path = storage.save_workflow_result(result)
events_path = storage.save_events_json(result)
summary_path = storage.save_summary_json(result)

# Evaluate summaries
evaluator = LegalSummarizationEvaluator()
metrics = evaluator.evaluate_summary(
    generated_summary=result.generated_summary,
    reference_text="Reference summary or key sections",
    original_document=result.document_text,
    legal_events=result.major_events
)

print(f"ROUGE-2: {metrics.rouge_2}")
print(f"Legal Accuracy: {metrics.legal_accuracy}")
```

## 🔧 Configuration

### Settings

The system uses a comprehensive configuration system. Key settings can be modified in `config/settings.py`:

```python
from legal_summarization.config import settings

# Modify Gemini settings
settings.gemini.temperature = 0.2
settings.gemini.max_tokens = 4096

# Modify workflow settings
settings.workflow.max_retries = 5
settings.workflow.enable_parallel_processing = True

# Modify evaluation settings
settings.evaluation.rouge_types = ["rouge1", "rouge2", "rougeL"]
settings.evaluation.enable_bertscore = True
```

### Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini LLM access
- `LOG_LEVEL`: Set logging level (default: INFO)
- `OUTPUT_DIR`: Directory for storing results (default: output)

## 📊 Output Format

### JSON Schema

The system generates structured JSON output following this schema:

```json
{
  "case_metadata": {
    "case_number": "string",
    "court": "string",
    "date": "string",
    "parties": ["string"]
  },
  "major_events": [
    {
      "event_id": "string",
      "title": "string",
      "description": "string",
      "timestamp": "string",
      "legal_significance": "string",
      "sub_events": [
        {
          "sub_event_id": "string",
          "description": "string",
          "evidence": ["string"],
          "legal_references": ["string"]
        }
      ]
    }
  ],
  "legal_reasoning": {
    "key_issues": ["string"],
    "precedents_cited": ["string"],
    "legal_principles": ["string"]
  },
  "summary_components": {
    "facts": "string",
    "issues": "string",
    "reasoning": "string",
    "conclusion": "string"
  }
}
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

### Automatic Metrics
- **ROUGE-1, ROUGE-2, ROUGE-L**: Lexical overlap metrics
- **BERTScore**: Semantic similarity metric
- **Legal Quality Metrics**: Domain-specific evaluation

### Legal Quality Dimensions
- **Legal Accuracy** (0-10): Correctness of legal facts and terminology
- **Completeness** (0-10): Coverage of all major legal points
- **Coherence** (0-10): Logical flow and structure
- **Clarity** (0-10): Readability and understandability
- **Legal Language** (0-10): Appropriate use of legal terminology

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest legal_summarization/tests/

# Run with coverage
pytest --cov=legal_summarization legal_summarization/tests/

# Run specific test categories
pytest legal_summarization/tests/test_workflow.py::TestLegalSummarizationWorkflow
```

### Test Coverage

The test suite covers:
- State management and serialization
- Individual agent functionality
- Workflow orchestration
- JSON storage and retrieval
- Evaluation metrics
- Preprocessing utilities
- Integration scenarios

## 🔍 Example Output

### Sample Input
```
IN THE HIGH COURT OF JAMMU AND KASHMIR
Case No: WP(C) 456/2022
M/s XYZ Industries vs State of J&K

The petitioner challenges the government order...
```

### Sample Output
```json
{
  "case_metadata": {
    "case_number": "WP(C) 456/2022",
    "court": "High Court of Jammu and Kashmir",
    "parties": ["M/s XYZ Industries", "State of J&K"]
  },
  "generated_summary": "The High Court dismissed the writ petition filed by XYZ Industries challenging the government order. The court held that retrospective application of executive orders is not permissible...",
  "evaluation_scores": {
    "rouge_2": 0.65,
    "legal_accuracy": 8.5,
    "overall_score": 7.8
  }
}
```

## 🛠️ Advanced Features

### Custom Agents

Extend the system with custom agents:

```python
from legal_summarization.agents.preprocessor import DocumentPreprocessorAgent

class CustomPreprocessorAgent(DocumentPreprocessorAgent):
    async def process(self, state):
        # Custom preprocessing logic
        state = await super().process(state)
        # Additional custom processing
        return state
```

### Custom Evaluation

Implement domain-specific evaluation:

```python
from legal_summarization.utils.evaluation import LegalSummarizationEvaluator

class CustomEvaluator(LegalSummarizationEvaluator):
    def _calculate_legal_metrics(self, generated, reference, original, events):
        # Custom legal evaluation logic
        return super()._calculate_legal_metrics(generated, reference, original, events)
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Retry Logic**: Automatic retries for transient failures
- **Graceful Degradation**: Fallback methods when LLM calls fail
- **Error Propagation**: Detailed error messages in workflow state
- **Logging**: Comprehensive logging for debugging

## 📚 Dependencies

### Core Dependencies
- `langchain>=0.0.350` - LangChain framework
- `langgraph>=0.0.60` - LangGraph for workflow orchestration
- `google-generativeai>=0.3.0` - Gemini LLM integration
- `pydantic>=2.0.0` - Data validation and serialization

### Evaluation Dependencies
- `rouge-score>=0.1.2` - ROUGE metrics
- `bert-score>=0.3.13` - BERTScore evaluation
- `nltk>=3.8.1` - Natural language processing

### Development Dependencies
- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black

# Run tests
pytest

# Format code
black legal_summarization/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of the MILDSum dataset for Indian legal document summarization
- Uses Google's Gemini LLM for advanced language understanding
- Leverages LangGraph for sophisticated workflow orchestration
- Incorporates ROUGE and BERTScore for comprehensive evaluation

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](https://github.com/your-repo/legal-summarization/issues) page
2. Review the [Documentation](https://github.com/your-repo/legal-summarization/wiki)
3. Contact the maintainers

---

**Note**: This system is designed for research and educational purposes. For production legal applications, ensure compliance with relevant legal and ethical guidelines.