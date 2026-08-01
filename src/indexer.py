import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class FinancialTableIndexer:
    """
    Indexes extracted financial report tables by Ticker, Year, Scope, Document ID, and Table Page/Position.
    """

    def __init__(self, json_tables_path: str = "all_financial_tables.json"):
        self.json_tables_path = json_tables_path
        self.doc_index: Dict[str, Dict[str, Any]] = {}
        self.tables_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        self.ticker_year_index: Dict[str, Dict[int, Dict[str, List[str]]]] = {}
        self.is_indexed = False

    def build_index(self):
        """
        Builds in-memory index from extracted tables JSON.
        """
        if self.is_indexed:
            return

        if not os.path.exists(self.json_tables_path):
            print(f"Warning: Extracted tables file {self.json_tables_path} not found.")
            return

        print(f"Indexing financial tables from {self.json_tables_path}...")
        with open(self.json_tables_path, "r", encoding="utf-8") as f:
            raw_tables = json.load(f)

        for tbl in raw_tables:
            path_file = tbl.get("path_file", "")
            doc_id = self.extract_doc_id(path_file)
            if not doc_id:
                continue

            ticker, year, scope = self.parse_doc_id(doc_id)
            page = str(tbl.get("page", "")).strip()

            table_entry = {
                "doc_id": doc_id,
                "page": page,
                "table_key": f"{doc_id}|{page}",
                "report_type": tbl.get("report_type", ""),
                "company": tbl.get("company", ""),
                "year": year,
                "scope": scope,
                "table_data": tbl.get("table_data", [])
            }

            if doc_id not in self.tables_by_doc:
                self.tables_by_doc[doc_id] = []
            self.tables_by_doc[doc_id].append(table_entry)

            # Build ticker -> year -> scope -> doc_ids index
            if ticker not in self.ticker_year_index:
                self.ticker_year_index[ticker] = {}
            if year not in self.ticker_year_index[ticker]:
                self.ticker_year_index[ticker][year] = {"consolidated": [], "separate": [], "any": []}

            if doc_id not in self.ticker_year_index[ticker][year][scope]:
                self.ticker_year_index[ticker][year][scope].append(doc_id)
            if doc_id not in self.ticker_year_index[ticker][year]["any"]:
                self.ticker_year_index[ticker][year]["any"].append(doc_id)

        self.is_indexed = True
        print(f"Indexed {len(self.tables_by_doc)} documents across {len(self.ticker_year_index)} tickers.")

    @staticmethod
    def extract_doc_id(path_file: str) -> str:
        """
        Extracts document ID from path.
        Example: ViFinQA/financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated/AAA_financial_statements_2015_consolidated_extracted.txt
        Output: AAA_financial_statements_2015_consolidated
        """
        if not path_file:
            return ""
        norm_path = path_file.replace("\\", "/")
        filename = norm_path.split("/")[-1]
        if filename.endswith("_extracted.txt"):
            return filename[:-len("_extracted.txt")]
        elif filename.endswith(".txt"):
            return filename[:-4]
        parent_dir = norm_path.split("/")[-2] if len(norm_path.split("/")) > 1 else filename
        return parent_dir

    @staticmethod
    def parse_doc_id(doc_id: str):
        """
        Parses doc_id like AAA_financial_statements_2015_consolidated into (ticker, year, scope).
        """
        parts = doc_id.split("_")
        ticker = parts[0].upper() if parts else "UNKNOWN"
        year = 0
        scope = "consolidated"

        for p in parts:
            if p.isdigit() and len(p) == 4:
                year = int(p)
            elif "separate" in p.lower() or "parent" in p.lower() or "rieng" in p.lower():
                scope = "separate"
            elif "consolidated" in p.lower() or "hopnhat" in p.lower():
                scope = "consolidated"

        return ticker, year, scope

    def get_candidate_docs(self, tickers: List[str], years: List[int], scope: str = "any") -> List[str]:
        """
        Returns relevant doc_ids matching tickers, years, and scope.
        """
        candidate_docs = []
        if not self.is_indexed:
            self.build_index()

        target_tickers = tickers if tickers else list(self.ticker_year_index.keys())

        for ticker in target_tickers:
            t_upper = ticker.upper()
            if t_upper not in self.ticker_year_index:
                continue
            
            ticker_dict = self.ticker_year_index[t_upper]
            target_years = years if years else list(ticker_dict.keys())

            for y in target_years:
                if y in ticker_dict:
                    if scope == "parent" or scope == "separate":
                        docs = ticker_dict[y].get("separate", []) or ticker_dict[y].get("any", [])
                    elif scope == "consolidated":
                        docs = ticker_dict[y].get("consolidated", []) or ticker_dict[y].get("any", [])
                    else:
                        docs = ticker_dict[y].get("any", [])
                    candidate_docs.extend(docs)

        return list(dict.fromkeys(candidate_docs))
