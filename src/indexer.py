import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class FinancialTableIndexer:
    """
    Indexes financial report tables locally by Ticker, Year, Scope (consolidated/separate),
    Document ID, and Table Reference / Page.
    Supports both processed_data/table_catalog.json and all_financial_tables.json.
    """

    def __init__(self, json_tables_path: Optional[str] = None):
        # Auto-detect default table catalog if not provided or not existing
        if not json_tables_path or not os.path.exists(json_tables_path):
            if os.path.exists("processed_data/table_catalog.json"):
                self.json_tables_path = "processed_data/table_catalog.json"
            elif os.path.exists("all_financial_tables.json"):
                self.json_tables_path = "all_financial_tables.json"
            elif os.path.exists("all_tables_structured.json"):
                self.json_tables_path = "all_tables_structured.json"
            else:
                self.json_tables_path = json_tables_path or "processed_data/table_catalog.json"
        else:
            self.json_tables_path = json_tables_path

        self.tables_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        self.ticker_year_index: Dict[str, Dict[int, Dict[str, List[str]]]] = {}
        self.is_indexed = False

    def build_index(self):
        """
        Builds in-memory index from the local catalog file.
        """
        if self.is_indexed:
            return

        if not os.path.exists(self.json_tables_path):
            print(f"Warning: Table catalog file '{self.json_tables_path}' not found.")
            return

        print(f"Building local table index from '{self.json_tables_path}'...")
        with open(self.json_tables_path, "r", encoding="utf-8") as f:
            raw_tables = json.load(f)

        for tbl in raw_tables:
            # Format A: table_catalog.json
            if "table_ref" in tbl and "report_id" in tbl:
                doc_id = tbl.get("report_id", "")
                ticker = str(tbl.get("ticker", "")).strip().upper()
                try:
                    year = int(tbl.get("year", 0))
                except Exception:
                    year = 0
                scope = str(tbl.get("report_type", "consolidated")).strip().lower()
                table_ref = tbl.get("table_ref", "")
                csv_filename = tbl.get("csv_filename", "")

                table_entry = {
                    "doc_id": doc_id,
                    "page": str(tbl.get("start_line", "")),
                    "table_key": table_ref,
                    "report_type": scope,
                    "ticker": ticker,
                    "year": year,
                    "scope": scope,
                    "csv_filename": csv_filename,
                    "headers": tbl.get("headers", []),
                    "sample_row_labels": tbl.get("sample_row_labels", []),
                    "table_data": tbl.get("table_data", [])
                }

            # Format B: all_financial_tables.json
            else:
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
                    "report_type": tbl.get("report_type", scope),
                    "company": tbl.get("company", ""),
                    "ticker": ticker,
                    "year": year,
                    "scope": scope,
                    "csv_filename": f"{doc_id}_table_{page}.csv",
                    "headers": [],
                    "sample_row_labels": [],
                    "table_data": tbl.get("table_data", [])
                }

            if not doc_id:
                continue

            if doc_id not in self.tables_by_doc:
                self.tables_by_doc[doc_id] = []
            self.tables_by_doc[doc_id].append(table_entry)

            # Build ticker -> year -> scope -> doc_ids index
            if ticker not in self.ticker_year_index:
                self.ticker_year_index[ticker] = {}
            if year not in self.ticker_year_index[ticker]:
                self.ticker_year_index[ticker][year] = {"consolidated": [], "separate": [], "any": []}

            scope_key = "separate" if scope in ["separate", "parent"] else "consolidated"
            if doc_id not in self.ticker_year_index[ticker][year][scope_key]:
                self.ticker_year_index[ticker][year][scope_key].append(doc_id)
            if doc_id not in self.ticker_year_index[ticker][year]["any"]:
                self.ticker_year_index[ticker][year]["any"].append(doc_id)

        self.is_indexed = True
        print(f"Indexed {len(self.tables_by_doc)} reports across {len(self.ticker_year_index)} tickers.")

    @staticmethod
    def extract_doc_id(path_file: str) -> str:
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
        candidate_docs = []
        if not self.is_indexed:
            self.build_index()

        target_tickers = [t.upper() for t in tickers] if tickers else list(self.ticker_year_index.keys())

        for ticker in target_tickers:
            if ticker not in self.ticker_year_index:
                continue

            ticker_dict = self.ticker_year_index[ticker]
            target_years = years if years else list(ticker_dict.keys())

            for y in target_years:
                if y in ticker_dict:
                    if scope in ["parent", "separate"]:
                        docs = ticker_dict[y].get("separate", []) or ticker_dict[y].get("any", [])
                    elif scope == "consolidated":
                        docs = ticker_dict[y].get("consolidated", []) or ticker_dict[y].get("any", [])
                    else:
                        docs = ticker_dict[y].get("any", [])
                    candidate_docs.extend(docs)

        # Fallback: if no candidates found with year filter, fetch all years for ticker
        if not candidate_docs and tickers:
            for ticker in target_tickers:
                if ticker in self.ticker_year_index:
                    for y_dict in self.ticker_year_index[ticker].values():
                        candidate_docs.extend(y_dict.get("any", []))

        return list(dict.fromkeys(candidate_docs))
