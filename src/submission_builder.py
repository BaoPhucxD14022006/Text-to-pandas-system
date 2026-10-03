import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from retrieval_engine import TableRetriever
    from pandas_generator import LLMPandasGenerator, generate_heuristic_pandas_query
    from executor import execute_pandas_query
except ImportError:
    from src.retrieval_engine import TableRetriever
    from src.pandas_generator import LLMPandasGenerator, generate_heuristic_pandas_query
    from src.executor import execute_pandas_query


class SubmissionBuilder:
    """
    Builds submission.json and packages referenced evidence CSVs into submission.zip.
    """

    def __init__(self, output_dir: str = "submission"):
        self.output_dir = output_dir
        self.sub_data_dir = os.path.join(output_dir, "data")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sub_data_dir, exist_ok=True)

    def build_submission(self, predictions: List[Dict[str, Any]], zip_filepath: str = "submission.zip"):
        # 1. Save submission.json
        submission_json_path = os.path.join(self.output_dir, "submission.json")
        with open(submission_json_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Saved submission.json to {submission_json_path}")

        # 2. Collect referenced CSV files
        used_csv_files = set()
        for p in predictions:
            for ev in p.get("evidence", []):
                rel_path = ev.get("csv_path", "")
                if rel_path:
                    used_csv_files.add(os.path.basename(rel_path))

        # 3. Create submission.zip
        print(f"Compressing {len(predictions)} records and {len(used_csv_files)} tables into {zip_filepath}...")
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(submission_json_path, arcname="submission.json")
            for csv_name in used_csv_files:
                local_csv = os.path.join(self.sub_data_dir, csv_name)
                # Copy from processed_data/csv if not already in submission/data
                if not os.path.exists(local_csv) and os.path.exists(os.path.join("processed_data/csv", csv_name)):
                    shutil.copy2(os.path.join("processed_data/csv", csv_name), local_csv)
                if os.path.exists(local_csv):
                    zipf.write(local_csv, arcname=f"data/{csv_name}")

        print(f"Successfully generated {zip_filepath}!")


def build_submission(questions_jsonl, catalog_path, stock_csv_path, csv_dir, output_dir, use_llm=False, limit=None, top_k=3):
    os.makedirs(output_dir, exist_ok=True)
    sub_data_dir = os.path.join(output_dir, "data")
    os.makedirs(sub_data_dir, exist_ok=True)

    print("Initializing Table Retriever...")
    retriever = TableRetriever(catalog_path, stock_csv_path)

    llm_gen = None
    if use_llm:
        llm_gen = LLMPandasGenerator()
        llm_gen.load_model()

    questions = []
    with open(questions_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if limit is not None and limit > 0:
        questions = questions[:limit]
        print(f"Limiting execution to first {len(questions)} questions.")
    else:
        print(f"Total questions to process: {len(questions)}")

    submission_records = []
    used_csv_files = set()

    for idx, q_item in enumerate(questions, 1):
        q_id = q_item['id']
        question_text = q_item['question']

        ret_res = retriever.retrieve_tables(question_text, top_k=top_k)
        relevant_docs = ret_res['relevant_docs']
        relevant_tables = ret_res['relevant_tables']
        top_candidates = ret_res['top_candidates']

        if use_llm and llm_gen:
            gen_res = llm_gen.generate_query(question_text, top_candidates, csv_dir)
        else:
            gen_res = generate_heuristic_pandas_query(question_text, top_candidates, csv_dir)

        pandas_query = gen_res['pandas_query']
        evidence = gen_res['evidence']

        answer, success = execute_pandas_query(pandas_query, evidence, csv_dir)

        for ev in evidence:
            csv_rel = ev['csv_path']
            csv_name = os.path.basename(csv_rel)
            used_csv_files.add(csv_name)

        record = {
            "id": q_id,
            "question": question_text,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "relevant_tables": relevant_tables,
            "evidence": evidence,
            "pandas_query": pandas_query
        }
        submission_records.append(record)

        if idx % 100 == 0 or idx == len(questions):
            print(f"Processed {idx}/{len(questions)} questions...")

    builder = SubmissionBuilder(output_dir=output_dir)
    workspace_dir = os.path.dirname(os.path.abspath(output_dir))
    zip_path = os.path.join(workspace_dir, "submission.zip")
    builder.build_submission(submission_records, zip_filepath=zip_path)


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    questions_jsonl = str(project_root / "ViFinQA" / "questions" / "questions.jsonl")
    catalog_path = str(project_root / "processed_data" / "table_catalog.json")
    stock_csv_path = str(project_root / "ViFinQA" / "code_stock.csv")
    csv_dir = str(project_root / "processed_data" / "csv")
    output_dir = str(project_root / "submission")

    build_submission(questions_jsonl, catalog_path, stock_csv_path, csv_dir, output_dir, use_llm=False)
