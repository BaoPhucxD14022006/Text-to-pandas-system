import os
import sys
import time
import argparse

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from table_parser import process_all_reports
from submission_builder import build_submission

def parse_args():
    parser = argparse.ArgumentParser(description="AI Guru Contest: ViFinQA Text-to-Pandas Pipeline")
    parser.add_argument("--use-llm", action="store_true", default=False, help="Enable LLM for Python Pandas code generation")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions (for testing/debugging)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate tables to retrieve per question (default: 3)")
    parser.add_argument("--force-reparse", action="store_true", default=False, help="Force re-parsing of OCR HTML tables even if catalog exists")
    return parser.parse_args()

def main():
    args = parse_args()

    print("==================================================")
    print("   AI_GURU CONTEST: TEXT-TO-PANDAS PIPELINE     ")
    print("==================================================")
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    statements_dir = os.path.join(workspace_dir, "ViFinQA", "financial_statements")
    questions_jsonl = os.path.join(workspace_dir, "ViFinQA", "questions", "questions.jsonl")
    stock_csv_path = os.path.join(workspace_dir, "ViFinQA", "code_stock.csv")
    
    processed_dir = os.path.join(workspace_dir, "processed_data")
    csv_dir = os.path.join(processed_dir, "csv")
    catalog_path = os.path.join(processed_dir, "table_catalog.json")
    
    output_dir = os.path.join(workspace_dir, "submission")

    t0 = time.time()

    # Step 1: Data Parsing & Table Normalization
    print("\n--- STEP 1: PARSING & CLEANING OCR TABLES ---")
    if not os.path.exists(catalog_path) or args.force_reparse:
        process_all_reports(statements_dir, csv_dir, catalog_path)
    else:
        print(f"Catalog already exists at {catalog_path}. Skipping extraction (use --force-reparse to re-run).")

    # Step 2-6: Retrieval, Code Generation, Execution & Packaging
    print("\n--- STEP 2-6: RETRIEVAL, CODE GEN & SUBMISSION PACKAGING ---")
    build_submission(
        questions_jsonl=questions_jsonl,
        catalog_path=catalog_path,
        stock_csv_path=stock_csv_path,
        csv_dir=csv_dir,
        output_dir=output_dir,
        use_llm=args.use_llm,
        limit=args.limit,
        top_k=args.top_k
    )

    t1 = time.time()
    print(f"\nPipeline completed successfully in {t1 - t0:.2f} seconds!")
    print(f"Submission package generated at: {os.path.join(workspace_dir, 'submission.zip')}")

if __name__ == '__main__':
    main()
