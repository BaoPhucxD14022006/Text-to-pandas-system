import os
import sys
import json
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.indexer import FinancialTableIndexer
from src.llm_client import LLMClient
from src.stage1_v2equery import Stage1QueryParser
from src.retriever import FinancialRetriever
from src.stage2_t2pandas import Stage2PandasGenerator


def test_question(
    question_text: str,
    question_id: int = 1,
    server_url: str = None,
    tables_path: str = "processed_data/table_catalog.json",
    csv_dir: str = "processed_data/csv",
    stock_csv_path: str = "ViFinQA/code_stock.csv",
    output_dir: str = "submission"
):
    print("\n" + "=" * 60)
    print(f" TESTING QUESTION (ID: {question_id})")
    print(f" Question: {question_text}")
    print("=" * 60)

    # 1. Init local components
    indexer = FinancialTableIndexer(json_tables_path=tables_path)
    indexer.build_index()

    use_llm = bool(server_url)
    llm_client = LLMClient(base_url=server_url, enabled=use_llm)
    stage1_parser = Stage1QueryParser(stock_csv_path=stock_csv_path, llm_client=llm_client)
    retriever = FinancialRetriever(
        indexer=indexer,
        csv_dir=csv_dir,
        output_data_dir=os.path.join(output_dir, "data")
    )
    stage2_generator = Stage2PandasGenerator(llm_client=llm_client)

    # 2. Stage 1: Vietnamese Query -> Canonical English Query
    print("\n [STAGE 1] Parsing Query & Entities...")
    parsed_intent = stage1_parser.parse(question_text)
    print(f"  - Tickers: {parsed_intent.get('tickers')}")
    print(f"  - Years: {parsed_intent.get('years')}")
    print(f"  - Scope: {parsed_intent.get('scope')}")
    print(f"  - Target Unit: {parsed_intent.get('target_unit')}")
    print(f"  - Financial Metrics: {parsed_intent.get('financial_metrics')}")
    print(f"  - Canonical Query: {parsed_intent.get('canonical_english_query')}")

    # 3. Retrieval: Document & Table Grounding
    print("\n [RETRIEVAL] Finding Relevant Documents & Tables...")
    evidence_res = retriever.retrieve(parsed_intent)
    print(f"  - Relevant Docs: {evidence_res.get('relevant_docs')}")
    print(f"  - Relevant Tables: {evidence_res.get('relevant_tables')}")
    print(f"  - Evidence CSVs: {[e['csv_path'] for e in evidence_res.get('evidence', [])]}")

    # 4. Stage 2: Pandas Code Generation & Execution
    print("\n [STAGE 2] Generating & Executing Pandas Query...")
    pandas_query, answer = stage2_generator.generate_and_execute(
        question=question_text,
        canonical_query=parsed_intent.get("canonical_english_query", ""),
        target_unit=parsed_intent.get("target_unit", "đồng"),
        dfs=evidence_res.get("dfs", {})
    )

    print("\n [FINAL OUTPUT]")
    print(f"  - Pandas Query: {pandas_query}")
    print(f"  - Calculated Answer: {answer}")
    print("=" * 60 + "\n")

    result_obj = {
        "id": question_id,
        "question": question_text,
        "answer": answer,
        "relevant_docs": evidence_res.get("relevant_docs", []),
        "relevant_tables": evidence_res.get("relevant_tables", []),
        "evidence": evidence_res.get("evidence", []),
        "pandas_query": pandas_query
    }

    print(json.dumps(result_obj, ensure_ascii=False, indent=2))
    return result_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single financial question locally")
    parser.add_argument("--question", type=str, help="Vietnamese question string to test")
    parser.add_argument("--id", type=int, default=1, help="Question ID in dataset")
    parser.add_argument("--tables", type=str, default="processed_data/table_catalog.json", help="Path to tables catalog")
    parser.add_argument("--csv-dir", type=str, default="processed_data/csv", help="Path to CSV tables")
    parser.add_argument("--server-url", type=str, default=None, help="Optional remote/local LLM Server URL")

    args = parser.parse_args()

    q_text = args.question
    if not q_text and os.path.exists("ViFinQA/questions/questions.jsonl"):
        with open("ViFinQA/questions/questions.jsonl", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                if data.get("id") == args.id or (idx + 1) == args.id:
                    q_text = data.get("question")
                    break

    if not q_text:
        q_text = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"

    test_question(
        question_text=q_text,
        question_id=args.id,
        server_url=args.server_url,
        tables_path=args.tables,
        csv_dir=args.csv_dir
    )
