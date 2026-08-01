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


def test_question(question_text: str, question_id: int = 1, server_url: str = None):
    print("\n" + "=" * 60)
    print(f" TESTING QUESTION (ID: {question_id})")
    print(f" Question: {question_text}")
    print("=" * 60)

    # 1. Base URL override if provided
    if server_url:
        os.environ["LLM_BASE_URL"] = server_url
        print(f"Connecting to LLM Server: {server_url}")

    # 2. Init components
    indexer = FinancialTableIndexer(json_tables_path="all_financial_tables.json")
    indexer.build_index()

    llm_client = LLMClient(base_url=server_url)
    stage1_parser = Stage1QueryParser(stock_csv_path="ViFinQA/code_stock.csv", llm_client=llm_client)
    retriever = FinancialRetriever(indexer=indexer, output_data_dir="output/data")
    stage2_generator = Stage2PandasGenerator(llm_client=llm_client)

    # 3. Stage 1: Vietnamese Query -> Canonical English Query
    print("\n [STAGE 1] Parsing Query with Qwen...")
    parsed_intent = stage1_parser.parse(question_text)
    print(f"  - Tickers: {parsed_intent.get('tickers')}")
    print(f"  - Years: {parsed_intent.get('years')}")
    print(f"  - Scope: {parsed_intent.get('scope')}")
    print(f"  - Target Unit: {parsed_intent.get('target_unit')}")
    print(f"  - Financial Metrics: {parsed_intent.get('financial_metrics')}")
    print(f"  - Canonical Query: {parsed_intent.get('canonical_english_query')}")

    # 4. Retrieval: Document & Table Grounding
    print("\n [RETRIEVAL] Finding Relevant Documents & Tables...")
    evidence_res = retriever.retrieve(parsed_intent)
    print(f"  - Relevant Docs: {evidence_res.get('relevant_docs')}")
    print(f"  - Relevant Tables: {evidence_res.get('relevant_tables')}")
    print(f"  - Evidence CSVs: {[e['csv_path'] for e in evidence_res.get('evidence', [])]}")

    # 5. Stage 2: DeepSeek Pandas Code Generation & Execution
    print("\n [STAGE 2] Generating & Executing Pandas Query with DeepSeek...")
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
    parser = argparse.ArgumentParser(description="Test single financial question")
    parser.add_argument("--question", type=str, help="Vietnamese question string to test")
    parser.add_argument("--id", type=int, default=1, help="Question ID in dataset")
    parser.add_argument("--server-url", type=str, default=None, help="Kaggle/Remote LLM Server ngrok URL")

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

    test_question(q_text, question_id=args.id, server_url=args.server_url)
