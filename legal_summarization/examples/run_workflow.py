"""Example script for running the legal summarization workflow."""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the legal_summarization module
sys.path.insert(0, str(Path(__file__).parent.parent))

from legal_summarization import LegalSummarizationWorkflow, GeminiClient
from legal_summarization.utils import JSONStorageManager, LegalSummarizationEvaluator


async def main():
    """Main example function."""
    # Check if we have the sample data
    sample_file = Path(__file__).parent.parent / "Data" / "MILDSum_Samples" / "Sample_2" / "EN_Judgment.txt"
    
    if not sample_file.exists():
        print("Sample data not found. Please ensure the MILDSum dataset is available.")
        return
    
    # Read sample judgment
    print("Reading sample legal judgment...")
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_judgment = f.read()
    
    print(f"Sample judgment loaded ({len(sample_judgment)} characters)")
    
    # Initialize workflow
    print("Initializing legal summarization workflow...")
    
    # Note: You need to set GOOGLE_API_KEY environment variable
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable not set.")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        return
    
    try:
        workflow = LegalSummarizationWorkflow()
        
        # Process the document
        print("Processing legal document through workflow...")
        result = await workflow.process_document(
            document_text=sample_judgment,
            document_id="sample_2_demo",
            target_length=400
        )
        
        # Display results
        print("\n" + "="*50)
        print("WORKFLOW RESULTS")
        print("="*50)
        
        print(f"\nDocument ID: {result.document_id}")
        print(f"Processing Stage: {result.processing_stage}")
        print(f"Retry Count: {result.retry_count}")
        
        if result.error_messages:
            print(f"\nErrors encountered: {len(result.error_messages)}")
            for error in result.error_messages:
                print(f"  - {error}")
        
        # Display case metadata
        if result.case_metadata:
            print(f"\nCase Metadata:")
            print(f"  Case Number: {result.case_metadata.case_number}")
            print(f"  Court: {result.case_metadata.court}")
            print(f"  Date: {result.case_metadata.date}")
            print(f"  Parties: {', '.join(result.case_metadata.parties)}")
        
        # Display major events
        print(f"\nMajor Events Extracted: {len(result.major_events)}")
        for i, event in enumerate(result.major_events[:5], 1):
            print(f"  {i}. {event.title} ({event.event_type})")
            print(f"     Significance: {event.legal_significance}")
            if event.sub_events:
                print(f"     Sub-events: {len(event.sub_events)}")
        
        # Display legal reasoning
        if result.legal_reasoning:
            print(f"\nLegal Reasoning:")
            print(f"  Key Issues: {len(result.legal_reasoning.key_issues)}")
            for issue in result.legal_reasoning.key_issues[:3]:
                print(f"    - {issue}")
            
            print(f"  Precedents Cited: {len(result.legal_reasoning.precedents_cited)}")
            for precedent in result.legal_reasoning.precedents_cited[:3]:
                print(f"    - {precedent}")
        
        # Display generated summary
        if result.generated_summary:
            print(f"\nGenerated Summary ({len(result.generated_summary.split())} words):")
            print("-" * 50)
            print(result.generated_summary)
            print("-" * 50)
        
        # Display evaluation scores
        if result.evaluation_scores:
            print(f"\nEvaluation Scores:")
            print(f"  ROUGE-1: {result.evaluation_scores.rouge_1:.3f}")
            print(f"  ROUGE-2: {result.evaluation_scores.rouge_2:.3f}")
            print(f"  ROUGE-L: {result.evaluation_scores.rouge_l:.3f}")
            print(f"  BERTScore: {result.evaluation_scores.bert_score:.3f}")
            
            if result.evaluation_scores.legal_metrics:
                metrics = result.evaluation_scores.legal_metrics
                print(f"  Legal Accuracy: {metrics.legal_accuracy:.1f}/10")
                print(f"  Completeness: {metrics.completeness:.1f}/10") 
                print(f"  Coherence: {metrics.coherence:.1f}/10")
                print(f"  Clarity: {metrics.clarity:.1f}/10")
                print(f"  Overall Score: {metrics.overall_score:.1f}/10")
        
        # Save results
        print("\nSaving results...")
        storage = JSONStorageManager()
        
        # Save complete workflow result
        result_path = storage.save_workflow_result(result)
        print(f"Complete result saved to: {result_path}")
        
        # Save events JSON
        events_path = storage.save_events_json(result)
        print(f"Events JSON saved to: {events_path}")
        
        # Save summary JSON
        summary_path = storage.save_summary_json(result)
        print(f"Summary JSON saved to: {summary_path}")
        
        print("\nWorkflow completed successfully!")
        
    except Exception as e:
        print(f"Error running workflow: {str(e)}")
        import traceback
        traceback.print_exc()


def run_sync_example():
    """Example of synchronous workflow execution."""
    print("Running synchronous example...")
    
    # Simple text for testing
    sample_text = """
    IN THE HIGH COURT OF JAMMU AND KASHMIR

    Case No: WP(C) 123/2023
    
    M/s ABC Company vs State of J&K
    
    Facts: The petitioner company filed a writ petition challenging the 
    government order dated 03.03.2010 which imposed a ban on power 
    connections for industrial units.
    
    Issues: Whether the ban can be applied retrospectively to existing 
    power connections.
    
    Held: The court observed that retrospective operation of a government 
    order cannot be permitted where it is merely an executive order.
    
    Conclusion: The petition is dismissed.
    """
    
    try:
        workflow = LegalSummarizationWorkflow()
        
        # Process synchronously
        result = workflow.process_document_sync(
            document_text=sample_text,
            document_id="sync_example",
            target_length=150
        )
        
        print(f"Summary: {result.generated_summary}")
        print(f"Events: {len(result.major_events)}")
        
    except Exception as e:
        print(f"Synchronous example failed: {str(e)}")


if __name__ == "__main__":
    print("Legal Summarization Workflow Example")
    print("="*40)
    
    # Check if we're in an environment where we can run async
    try:
        # Run async example
        asyncio.run(main())
    except Exception as e:
        print(f"Async example failed: {str(e)}")
        print("\nTrying synchronous example instead...")
        run_sync_example()