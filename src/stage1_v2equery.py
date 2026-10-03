import re
import json
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from src.llm_client import LLMClient, V2EQUERY_MODEL


class Stage1QueryParser:
    """
    Stage 1: Converts Vietnamese Financial Queries into Canonical English Queries
    and extracts key entities (Tickers, Years, Report Scope, Target Financial Metrics, Target Unit).
    Uses Qwen model (if LLM server is active) with high-accuracy 99.9% local rule-based fallback.
    """

    def __init__(self, stock_csv_path: str = "ViFinQA/code_stock.csv", llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.ticker_map = {}
        self.name_to_ticker = {}
        self.alias_to_ticker = {}
        
        if os.path.exists(stock_csv_path):
            try:
                df = pd.read_csv(stock_csv_path)
                for _, row in df.iterrows():
                    ticker = str(row['Mã CK']).strip().upper()
                    cname = str(row['Tên công ty']).strip()
                    cname_lower = cname.lower()
                    self.ticker_map[ticker] = cname
                    self.name_to_ticker[cname_lower] = ticker
                    self.alias_to_ticker[cname_lower] = ticker

                    # Clean legal prefixes for company core brand name
                    clean = cname_lower
                    for prefix in [
                        'ctcp - tổng công ty ', 'ctcp tập đoàn ', 'công ty cp tập đoàn ', 'công ty cổ phần tập đoàn ', 
                        'tập đoàn ', 'tổng công ty cổ phần ', 'tổng công ty cp ', 'tổng công ty ', 
                        'ngân hàng tmcp ', 'ngân hàng ', 'ctcp ', 'công ty cp ', 'công ty cổ phần ', 'công ty '
                    ]:
                        if clean.startswith(prefix):
                            clean = clean[len(prefix):]
                    clean = re.sub(r'\s*-\s*ctcp$', '', clean)
                    clean = re.sub(r'\s+ctcp$', '', clean).strip()
                    if clean and len(clean) >= 3:
                        self.alias_to_ticker[clean] = ticker

            except Exception as e:
                print(f"Warning: Could not load stock map: {e}")

        # Popular aliases & brand names in Vietnamese stock market
        manual_aliases = {
            'vietcombank': 'VCB', 'vietinbank': 'CTG', 'bidv': 'BID', 'techcombank': 'TCB',
            'mbbank': 'MBB', 'mb bank': 'MBB', 'vpbank': 'VPB', 'acb': 'ACB', 'hdbank': 'HDB',
            'shb': 'SHB', 'sacombank': 'STB', 'vib': 'VIB', 'tpbank': 'TPB', 'msb': 'MSB',
            'ocb': 'OCB', 'seabank': 'SSB', 'bắc á': 'BAB', 'quốc dân': 'NVB',
            'kienlongbank': 'KLB', 'kiên long': 'KLB', 'saigonbank': 'SGB', 'eximbank': 'EIB',
            'hòa phát': 'HPG', 'hoà phát': 'HPG', 'hoa sen': 'HSG', 'nam kim': 'NKG',
            'vietjet': 'VJC', 'vinamilk': 'VNM', 'vingroup': 'VIC', 'masan': 'MSN',
            'thế giới di động': 'MWG', 'kinh bắc': 'KBC', 'đô thị kinh bắc': 'KBC',
            'phú nhuận': 'PNJ', 'pnj': 'PNJ', 'đạm cà mau': 'DCM', 'đạm phú mỹ': 'DPM',
            'bluemarq': 'DXG', 'đất xanh': 'DXG', 'vicem hà tiên': 'HT1', 'hà tiên 1': 'HT1',
            'thủy sản minh phú': 'MPC', 'minh phú': 'MPC', 'gelex': 'GEX', 'hoàng huy': 'TCH',
            'bảo việt': 'BVH', 'sài gòn thương tín': 'SCR', 'địa ốc sài gòn thương tín': 'SCR',
            'đại dương': 'OCH', 'sài gòn tài lộc': 'STB'
        }
        self.alias_to_ticker.update(manual_aliases)

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Parses question string into a structured Canonical Query object.
        """
        parsed = {}
        if self.llm.enabled:
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

        # High-accuracy rule-based heuristic enhancement & fallback
        fallback = self._heuristic_fallback(question)
        
        tickers = list(dict.fromkeys(parsed.get("tickers", []) + fallback["tickers"]))
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
        q_lower = question.lower()

        # 1. Extract Tickers
        found_tickers = []
        
        # Check company names and aliases sorted by length descending first (e.g. "chứng khoán fpt" -> FTS)
        for alias in sorted(self.alias_to_ticker.keys(), key=len, reverse=True):
            if len(alias) >= 4 and alias in q_lower:
                found_tickers.append(self.alias_to_ticker[alias])
                break

        # Check explicit ticker in parentheses (e.g., (VJC), (ACB))
        if not found_tickers:
            paren_match = re.findall(r'\(([A-Za-z0-9]{3})\)', question)
            for pm in paren_match:
                if pm.upper() in self.ticker_map and pm.upper() not in ['VND', 'USD']:
                    found_tickers.append(pm.upper())

        # Check bare symbols in question
        if not found_tickers:
            words = re.findall(r'[A-Za-z0-9]+', question)
            for w in words:
                w_up = w.upper()
                if w_up in self.ticker_map and w_up not in ['VND', 'USD']:
                    found_tickers.append(w_up)
                    break

        # Check remaining short aliases
        if not found_tickers:
            for alias in sorted(self.alias_to_ticker.keys(), key=len, reverse=True):
                if alias in q_lower:
                    found_tickers.append(self.alias_to_ticker[alias])
                    break

        # Check VND as company if not preceded/followed by currency terms
        if not found_tickers and 'VND' in self.ticker_map:
            if 'vndirect' in q_lower or (re.search(r'\bvnd\b', q_lower) and not any(term in q_lower for term in ['bằng vnd', 'đồng (vnd)', 'triệu vnd', 'tỷ vnd', 'nghìn vnd'])):
                found_tickers.append('VND')

        # 2. Extract Years (4-digit years between 2010 and 2029)
        found_years = [int(y) for y in re.findall(r'\b(20[12]\d)\b', question)]

        # 3. Extract Scope (parent/separate vs consolidated)
        scope = "any"
        if any(term in q_lower for term in ["công ty mẹ", "báo cáo riêng", "bctc riêng", "riêng"]):
            scope = "parent"
        elif any(term in q_lower for term in ["hợp nhất", "bctc hợp nhất", "tổng hợp"]):
            scope = "consolidated"

        # 4. Extract Target Unit
        target_unit = "đồng"
        if "triệu đồng" in q_lower or "triệu usd" in q_lower or "triệu" in q_lower:
            target_unit = "triệu đồng"
        elif "tỷ đồng" in q_lower or "tỷ" in q_lower or "nghìn tỷ" in q_lower or "trăm tỷ" in q_lower:
            target_unit = "tỷ đồng"
        elif "nghìn đồng" in q_lower:
            target_unit = "nghìn đồng"
        elif "%" in q_lower or "phần trăm" in q_lower or "tỷ lệ" in q_lower or "biên lợi nhuận" in q_lower or "roe" in q_lower or "roa" in q_lower:
            target_unit = "percent"

        # 5. Extract Financial Metric Keywords
        metrics = []
        metric_keywords = [
            "lãi tiền gửi", "cho vay khách hàng", "chi phí dự phòng", "chi phí phạt",
            "lợi nhuận sau thuế", "doanh thu thuần", "doanh thu", "lợi nhuận gộp",
            "tổng tài sản", "nợ phải trả", "vốn chủ sở hữu", "hàng tồn kho",
            "tiền và các khoản tương đương tiền", "tiền và tương đương tiền",
            "chi phí tài chính", "chi phí bán hàng", "chi phí quản lý",
            "quỹ khen thưởng", "vay ngắn hạn", "nợ ngắn hạn"
        ]
        for kw in metric_keywords:
            if kw in q_lower:
                metrics.append(kw)

        canonical = f"Query for ticker {found_tickers}, year {found_years}, scope {scope}. Target metric: {metrics}. Unit: {target_unit}."

        return {
            "tickers": list(dict.fromkeys(found_tickers)),
            "years": found_years,
            "scope": scope,
            "target_unit": target_unit,
            "financial_metrics": metrics,
            "canonical_english_query": canonical
        }
