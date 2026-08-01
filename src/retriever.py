import os
import json
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.indexer import FinancialTableIndexer


class FinancialRetriever:
    """
    Grounding & Information Retriever for ViFinQA.
    Retrieves relevant_docs, relevant_tables, generates evidence CSVs in data/,
    and returns DataFrames for Pandas code generation.
    """

    def __init__(self, indexer: FinancialTableIndexer, output_data_dir: str = "data"):
        self.indexer = indexer
        self.output_data_dir = output_data_dir
        os.makedirs(self.output_data_dir, exist_ok=True)

    def retrieve(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves grounded evidence for a query.
        Returns:
            - relevant_docs: list of string doc_ids
            - relevant_tables: list of string "doc_id|page"
            - evidence: list of dicts {"variable": "df1", "csv_path": "data/..."}
            - dfs: dict mapping variable name ("df1") to pandas DataFrame
        """
        tickers = parsed_query.get("tickers", [])
        years = parsed_query.get("years", [])
        scope = parsed_query.get("scope", "any")
        metrics = parsed_query.get("financial_metrics", [])

        candidate_docs = self.indexer.get_candidate_docs(tickers, years, scope)
        if not candidate_docs and tickers:
            # Fallback: relax scope to 'any'
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
                table_key = best_table["table_key"]
                relevant_tables.append(table_key)
                
                # Convert table_data to DataFrame
                raw_rows = best_table.get("table_data", [])
                if raw_rows:
                    df = pd.DataFrame(raw_rows)
                else:
                    df = pd.DataFrame()

                # Clean CSV filename
                clean_page = best_table['page'].replace(" ", "_").replace("-", "_")
                csv_filename = f"{doc_id}_table_{clean_page}.csv"
                csv_rel_path = f"data/{csv_filename}"
                csv_abs_path = os.path.join(self.output_data_dir, csv_filename)
                
                df.to_csv(csv_abs_path, index=False, encoding="utf-8-sig")

                var_name = f"df{var_idx}"
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
        Ranks tables within a document based on keyword overlap with metrics/headers.
        """
        if not tables:
            return None

        best_score = -1
        best_table = tables[0]

        metric_tokens = set([m.lower() for m in metrics])

        for tbl in tables:
            score = 0
            t_type = tbl.get("report_type", "").lower()
            
            # Boost main financial reports
            if "kết quả" in t_type or "cân đối" in t_type or "lưu chuyển" in t_type:
                score += 5

            rows = tbl.get("table_data", [])
            for row in rows:
                row_str = " ".join([str(v).lower() for v in row.values()])
                for token in metric_tokens:
                    if token in row_str:
                        score += 10
            
            if score > best_score:
                best_score = score
                best_table = tbl

        return best_table
