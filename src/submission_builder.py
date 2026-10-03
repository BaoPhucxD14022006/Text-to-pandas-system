import os
import json
import zipfile
import shutil
from pathlib import Path
import pandas as pd

try:
    from retrieval_engine import TableRetriever
    from pandas_generator import LLMPandasGenerator, generate_heuristic_pandas_query
    from executor import execute_pandas_query
except ImportError:
    from src.retrieval_engine import TableRetriever
    from src.pandas_generator import LLMPandasGenerator, generate_heuristic_pandas_query
    from src.executor import execute_pandas_query

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

        # 1. Retrieve relevant tables & docs
        ret_res = retriever.retrieve_tables(question_text, top_k=top_k)
        relevant_docs = ret_res['relevant_docs']
        relevant_tables = ret_res['relevant_tables']
        top_candidates = ret_res['top_candidates']

        # 2. Generate Pandas Query & Evidence
        if use_llm and llm_gen:
            gen_res = llm_gen.generate_query(question_text, top_candidates, csv_dir)
        else:
            gen_res = generate_heuristic_pandas_query(question_text, top_candidates, csv_dir)

        pandas_query = gen_res['pandas_query']
        evidence = gen_res['evidence']

        # 3. Execute query to compute float answer
        answer, success = execute_pandas_query(pandas_query, evidence, csv_dir)

        # Record used CSV files for zip packaging
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

    # Save submission.json
    submission_json_path = os.path.join(output_dir, "submission.json")
    with open(submission_json_path, 'w', encoding='utf-8') as f:
        json.dump(submission_records, f, ensure_ascii=False, indent=2)

    print(f"Saved submission.json to {submission_json_path}")

    # Copy referenced CSV files into sub_data_dir
    print(f"Copying {len(used_csv_files)} referenced CSV files to submission/data...")
    for csv_name in used_csv_files:
        src_csv = os.path.join(csv_dir, csv_name)
        dst_csv = os.path.join(sub_data_dir, csv_name)
        if os.path.exists(src_csv):
            shutil.copy2(src_csv, dst_csv)

    # Package into submission.zip
    workspace_dir = os.path.dirname(os.path.abspath(output_dir))
    zip_path = os.path.join(workspace_dir, "submission.zip")
    print(f"Compressing into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add submission.json
        zipf.write(submission_json_path, arcname="submission.json")
        # Add data/ CSV files
        for csv_name in used_csv_files:
            dst_csv = os.path.join(sub_data_dir, csv_name)
            if os.path.exists(dst_csv):
                zipf.write(dst_csv, arcname=f"data/{csv_name}")

    print(f"Successfully generated {zip_path}!")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    questions_jsonl = str(project_root / "ViFinQA" / "questions" / "questions.jsonl")
    catalog_path = str(project_root / "processed_data" / "table_catalog.json")
    stock_csv_path = str(project_root / "ViFinQA" / "code_stock.csv")
    csv_dir = str(project_root / "processed_data" / "csv")
    output_dir = str(project_root / "submission")

    build_submission(questions_jsonl, catalog_path, stock_csv_path, csv_dir, output_dir, use_llm=False)
