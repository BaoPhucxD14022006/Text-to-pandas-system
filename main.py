import os
import sys
import json
import argparse
from typing import List, Dict, Any

from src.indexer import FinancialTableIndexer
from src.llm_client import LLMClient
from src.stage1_v2equery import Stage1QueryParser
from src.retriever import FinancialRetriever
from src.stage2_t2pandas import Stage2PandasGenerator
from src.submission_builder import SubmissionBuilder

def load_questions(questions_path: str) -> List[Dict[str, Any]]:
    questions = []
    if not os.path.exists(questions_path):
        print(f"Error: Question file {questions_path} not found.")
        return questions

    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    questions.append(json.loads(line_str))
                except json.JSONDecodeError:
                    pass
    return questions


def run_pipeline(
    questions_path: str = "ViFinQA/questions/questions.jsonl",
    tables_path: str = "processed_data/table_catalog.json",
    csv_dir: str = "processed_data/csv",
    stock_csv_path: str = "ViFinQA/code_stock.csv",
    output_dir: str = "submission",
    zip_path: str = "submission.zip",
    limit: int = 0,
    server_url: str = None
):
    print("=== STARTING VIFINQA TEXT-TO-PANDAS PIPELINE ===")

    # 1. Load Questions
    questions = load_questions(questions_path)
    if limit > 0:
        questions = questions[:limit]
    print(f"Loaded {len(questions)} questions.")

    # 2. Initialize Components
    indexer = FinancialTableIndexer(json_tables_path=tables_path)
    indexer.build_index()

    llm_client = LLMClient(base_url=server_url, enabled=bool(server_url))
    stage1_parser = Stage1QueryParser(stock_csv_path=stock_csv_path, llm_client=llm_client)
    retriever = FinancialRetriever(
        indexer=indexer,
        csv_dir=csv_dir,
        output_data_dir=os.path.join(output_dir, "data")
    )
    stage2_generator = Stage2PandasGenerator(llm_client=llm_client)

    predictions = []

    # 3. Process Questions
    for idx, q in enumerate(questions):
        qid = q.get("id", idx + 1)
        question_text = q.get("question", "")

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Processing question [{idx + 1}/{len(questions)}]: ID {qid}")

        # Stage 1: Parse Query to Canonical English Query
        parsed_intent = stage1_parser.parse(question_text)

        # Retrieval: Find relevant_docs, relevant_tables, evidence CSVs
        retrieved_evidence = retriever.retrieve(parsed_intent)

        # Stage 2: Generate Pandas Code & Execute
        pandas_query, answer = stage2_generator.generate_and_execute(
            question=question_text,
            canonical_query=parsed_intent.get("canonical_english_query", ""),
            target_unit=parsed_intent.get("target_unit", "đồng"),
            dfs=retrieved_evidence.get("dfs", {})
        )

        prediction_item = {
            "id": qid,
            "question": question_text,
            "answer": answer,
            "relevant_docs": retrieved_evidence.get("relevant_docs", []),
            "relevant_tables": retrieved_evidence.get("relevant_tables", []),
            "evidence": retrieved_evidence.get("evidence", []),
            "pandas_query": pandas_query
        }
        predictions.append(prediction_item)

    # 4. Build Submission Package
    builder = SubmissionBuilder(output_dir=output_dir)
    builder.build_submission(predictions, zip_filepath=zip_path)

    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")


def main():
    parser = argparse.ArgumentParser(description="ViFinQA 2-Stage Text-to-Pandas Local Pipeline")
    parser.add_argument("--questions", type=str, default="ViFinQA/questions/questions.jsonl", help="Path to questions JSONL")
    parser.add_argument("--tables", type=str, default="processed_data/table_catalog.json", help="Path to table catalog JSON")
    parser.add_argument("--csv-dir", type=str, default="processed_data/csv", help="Path to table CSVs directory")
    parser.add_argument("--stock-csv", type=str, default="ViFinQA/code_stock.csv", help="Path to stock codes CSV")
    parser.add_argument("--output-dir", type=str, default="submission", help="Output directory")
    parser.add_argument("--zip-output", type=str, default="submission.zip", help="Path to submission.zip")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions to process (0 = all)")
    parser.add_argument("--server-url", type=str, default=None, help="Optional remote/local LLM server URL (e.g. Ollama or vLLM)")

    args = parser.parse_args()
    run_pipeline(
        questions_path=args.questions,
        tables_path=args.tables,
        csv_dir=args.csv_dir,
        stock_csv_path=args.stock_csv,
        output_dir=args.output_dir,
        zip_path=args.zip_output,
        limit=args.limit,
        server_url=args.server_url
    )


if __name__ == "__main__":
    main()
