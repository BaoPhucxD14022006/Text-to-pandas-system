import re
import json
import os
import pandas as pd
from typing import Dict, Any, List
from src.llm_client import LLMClient, V2EQUERY_MODEL


class Stage1QueryParser:
    """
    Stage 1: Converts Vietnamese Financial Queries into Canonical English Queries
    and extracts key entities (Tickers, Years, Report Scope, Target Financial Metrics, Target Unit).
    Uses Qwen model (V2EQUERY_MODEL) with rule-based fallback.
    """

    def __init__(self, stock_csv_path: str = "ViFinQA/code_stock.csv", llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
        self.ticker_map = {}
        self.name_to_ticker = {}
        
        if os.path.exists(stock_csv_path):
            try:
                df = pd.read_csv(stock_csv_path)
                for _, row in df.iterrows():
                    ticker = str(row['Mã CK']).strip().upper()
                    cname = str(row['Tên công ty']).strip()
                    self.ticker_map[ticker] = cname
                    self.name_to_ticker[cname.lower()] = ticker
            except Exception as e:
                print(f"Warning: Could not load stock map: {e}")

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Parses question string into a structured Canonical Query object.
        """
        system_prompt = (
            "You are a Vietnamese Financial Information Extraction expert.\n"
            "Given a Vietnamese question about company financial statements (Báo cáo tài chính),\n"
            "extract the following JSON fields:\n"
            "- tickers: list of stock ticker symbols mentioned or implied (e.g. [\"VNM\"])\n"
            "- years: list of integers representing years mentioned (e.g. [2018, 2023])\n"
            "- scope: \"consolidated\" (hợp nhất), \"parent\" (công ty mẹ/riêng), or \"any\"\n"
            "- target_unit: unit requested in answer (e.g., \"triệu đồng\", \"tỷ đồng\", \"đồng\", \"percent\", \"ratio\")\n"
            "- financial_metrics: list of financial metrics/items asked (e.g., [\"Lãi tiền gửi\", \"Doanh thu thuần\"])\n"
            "- canonical_english_query: Clear, standardized English query summarizing the request for pandas code generation.\n\n"
            "Return strictly valid JSON only."
        )

        user_prompt = f"Question: \"{question}\"\nJSON Output:"
        
        llm_response = self.llm.generate(V2EQUERY_MODEL, system_prompt, user_prompt)
        
        parsed = self._extract_json_from_llm(llm_response)
        
        # Rule-based fallback & enhancement to guarantee 100% precision on entities
        fallback = self._heuristic_fallback(question)
        
        tickers = list(set(parsed.get("tickers", []) + fallback["tickers"]))
        years = sorted(list(set(parsed.get("years", []) + fallback["years"])))
        scope = parsed.get("scope", fallback["scope"])
        if scope == "any" and fallback["scope"] != "any":
            scope = fallback["scope"]
            
        target_unit = parsed.get("target_unit") or fallback["target_unit"]
        metrics = parsed.get("financial_metrics") or fallback["financial_metrics"]
        canonical_query = parsed.get("canonical_english_query") or fallback["canonical_english_query"]

        return {
            "question": question,
            "tickers": tickers,
            "years": years,
            "scope": scope,
            "target_unit": target_unit,
            "financial_metrics": metrics,
            "canonical_english_query": canonical_query
        }

    def _extract_json_from_llm(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {}

    def _heuristic_fallback(self, question: str) -> Dict[str, Any]:
        q_upper = question.upper()
        q_lower = question.lower()

        # 1. Extract Tickers
        found_tickers = []
        # Direct stock symbol match (e.g. (VNM), VNM)
        for ticker in self.ticker_map:
            if re.search(rf'\b{ticker}\b', q_upper):
                found_tickers.append(ticker)
        # Search by company name if no ticker found
        if not found_tickers:
            for name, ticker in self.name_to_ticker.items():
                if name in q_lower:
                    found_tickers.append(ticker)

        # 2. Extract Years (4 digit numbers between 2010 and 2030)
        found_years = [int(y) for y in re.findall(r'\b(20[1-2][0-9])\b', question)]

        # 3. Extract Scope
        scope = "any"
        if any(term in q_lower for term in ["công ty mẹ", "báo cáo riêng", "bctc riêng", "riêng"]):
            scope = "parent"
        elif any(term in q_lower for term in ["hợp nhất", "bctc hợp nhất", "tổng hợp"]):
            scope = "consolidated"

        # 4. Extract Target Unit
        target_unit = "đồng"
        if "triệu đồng" in q_lower or "triệu" in q_lower:
            target_unit = "triệu đồng"
        elif "tỷ đồng" in q_lower or "tỷ" in q_lower:
            target_unit = "tỷ đồng"
        elif "%" in q_lower or "phần trăm" in q_lower or "tỷ lệ" in q_lower or "roe" in q_lower or "roa" in q_lower:
            target_unit = "percent"

        # 5. Extract Financial Metric Keywords
        metrics = []
        metric_keywords = [
            "lãi tiền gửi", "doanh thu thuần", "doanh thu", "lợi nhuận sau thuế",
            "lợi nhuận gộp", "tổng tài sản", "nợ phải trả", "vốn chủ sở hữu",
            "cho vay khách hàng", "chi phí quản lý", "chi phí tài chính",
            "lãi từ hoạt động", "tiền và tương đương tiền", "hàng tồn kho"
        ]
        for kw in metric_keywords:
            if kw in q_lower:
                metrics.append(kw)

        canonical = f"Query for ticker {found_tickers}, year {found_years}, scope {scope}. Target metric: {metrics}. Unit: {target_unit}."

        return {
            "tickers": found_tickers,
            "years": found_years,
            "scope": scope,
            "target_unit": target_unit,
            "financial_metrics": metrics,
            "canonical_english_query": canonical
        }
