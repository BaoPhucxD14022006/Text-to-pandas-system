import os
import json
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional
from src.indexer import FinancialTableIndexer


class FinancialRetriever:
    """
    Grounding & Information Retriever for ViFinQA (Local Execution).
    Retrieves relevant_docs, relevant_tables, generates evidence CSVs in output/data/,
    and returns DataFrames for Pandas code generation.
    """

    def __init__(
        self,
        indexer: FinancialTableIndexer,
        csv_dir: str = "processed_data/csv",
        output_data_dir: str = "submission/data"
    ):
        self.indexer = indexer
        self.csv_dir = csv_dir
        self.output_data_dir = output_data_dir
        os.makedirs(self.output_data_dir, exist_ok=True)

    def retrieve(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves grounded evidence for a query.
        Returns:
            - relevant_docs: list of string doc_ids
            - relevant_tables: list of string "doc_id|page" or table_key
            - evidence: list of dicts {"variable": "df1", "csv_path": "data/..."}
            - dfs: dict mapping variable name ("df1") to pandas DataFrame
        """
        tickers = parsed_query.get("tickers", [])
        years = parsed_query.get("years", [])
        scope = parsed_query.get("scope", "any")
        metrics = parsed_query.get("financial_metrics", [])

        candidate_docs = self.indexer.get_candidate_docs(tickers, years, scope)
        if not candidate_docs and tickers:
            candidate_docs = self.indexer.get_candidate_docs(tickers, years, "any")

        relevant_docs = candidate_docs[:2]  # Top candidate documents
        relevant_tables = []
        evidence = []
        dfs = {}

        var_idx = 1
        for doc_id in relevant_docs:
            tables = self.indexer.tables_by_doc.get(doc_id, [])
            best_table = self._rank_tables_by_relevance(tables, metrics)
            
            if best_table:
                table_key = best_table.get("table_key", f"{doc_id}|{best_table.get('page', '1')}")
                relevant_tables.append(table_key)
                
                csv_filename = best_table.get("csv_filename", "")
                if not csv_filename:
                    clean_page = str(best_table.get('page', '1')).replace(" ", "_").replace("-", "_")
                    csv_filename = f"{doc_id}_table_{clean_page}.csv"

                df = pd.DataFrame()
                src_csv = os.path.join(self.csv_dir, csv_filename)
                dst_csv = os.path.join(self.output_data_dir, csv_filename)

                # 1. Try reading from processed_data/csv
                if os.path.exists(src_csv):
                    try:
                        df = pd.read_csv(src_csv, dtype=str)
                        shutil.copy2(src_csv, dst_csv)
                    except Exception:
                        df = pd.DataFrame()

                # 2. Or build from in-memory table_data if available
                if df.empty and best_table.get("table_data"):
                    df = pd.DataFrame(best_table["table_data"])
                    df.to_csv(dst_csv, index=False, encoding="utf-8-sig")

                var_name = f"df{var_idx}"
                csv_rel_path = f"data/{csv_filename}"
                evidence.append({
                    "variable": var_name,
                    "csv_path": csv_rel_path
                })
                dfs[var_name] = df
                var_idx += 1

        return {
            "relevant_docs": relevant_docs,
            "relevant_tables": relevant_tables,
            "evidence": evidence,
            "dfs": dfs
        }

    def _rank_tables_by_relevance(self, tables: List[Dict[str, Any]], metrics: List[str]) -> Optional[Dict[str, Any]]:
        """
        Ranks tables within a document based on keyword overlap with metrics, headers, and row labels.
        """
        if not tables:
            return None

        best_score = -1
        best_table = tables[0]

        metric_tokens = set([m.lower() for m in metrics])

        for tbl in tables:
            score = 0
            t_type = tbl.get("report_type", "").lower()
            table_key = tbl.get("table_key", "").lower()
            
            # Boost main financial statements
            if any(k in t_type or k in table_key for k in ["kết quả", "cân đối", "lưu chuyển", "thu nhập"]):
                score += 15

            # Search in sample_row_labels
            sample_labels = tbl.get("sample_row_labels", [])
            for lbl in sample_labels:
                lbl_lower = str(lbl).lower()
                for token in metric_tokens:
                    if token in lbl_lower:
                        score += 20

            # Search in headers
            headers = tbl.get("headers", [])
            for h in headers:
                h_lower = str(h).lower()
                for token in metric_tokens:
                    if token in h_lower:
                        score += 10

            # Search in raw rows if table_data exists
            rows = tbl.get("table_data", [])
            for row in rows:
                row_str = " ".join([str(v).lower() for v in row.values()]) if isinstance(row, dict) else str(row).lower()
                for token in metric_tokens:
                    if token in row_str:
                        score += 20
            
            if score > best_score:
                best_score = score
                best_table = tbl

        return best_table
