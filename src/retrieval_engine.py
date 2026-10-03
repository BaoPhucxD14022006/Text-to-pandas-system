import os
import re
import json
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

def clean_tokens(text):
    """Tokenize and strip punctuation for BM25 matching."""
    tokens = re.findall(r'[\w]+', str(text).lower())
    return tokens

class TableRetriever:
    def __init__(self, catalog_path, stock_csv_path):
        print(f"Loading table catalog from {catalog_path}...")
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)

        print(f"Loading stock mappings from {stock_csv_path}...")
        self.stock_df = pd.read_csv(stock_csv_path)
        self.valid_tickers = set(self.stock_df['Mã CK'].dropna().astype(str).str.strip().str.upper())
        
        self.alias_to_ticker = {}
        for _, row in self.stock_df.iterrows():
            ticker = str(row['Mã CK']).strip().upper()
            full_name = str(row['Tên công ty']).strip().lower()
            self.alias_to_ticker[full_name] = ticker
            
            # Clean common organizational prefixes to get core company brand name
            clean = full_name
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

        # Popular aliases in Vietnamese financial markets
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

        self.catalog_index = {}
        for item in self.catalog:
            t = item['ticker'].upper()
            y = str(item['year'])
            rt = item['report_type'].lower()
            key = (t, y, rt)
            if key not in self.catalog_index:
                self.catalog_index[key] = []
            self.catalog_index[key].append(item)

        print("Building precomputed global BM25 index...")
        self.global_corpus = []
        for c in self.catalog:
            headers_text = " ".join(c['headers'])
            labels_text = " ".join(c['sample_row_labels'])
            doc_text = f"{c['ticker']} {c['year']} {c['report_type']} {headers_text} {labels_text}"
            self.global_corpus.append(clean_tokens(doc_text))
        self.global_bm25 = BM25Okapi(self.global_corpus)
        print("Global BM25 index ready.")

    def extract_metadata_from_question(self, question):
        q_lower = question.lower()
        
        # 1. Find ticker in question
        ticker = None
        # Check explicit symbols, in uppercase, lowercase or parentheses (e.g., VJC, (ACB), HT1, PC1, hpx)
        words = re.findall(r'[A-Za-z0-9]+', question)
        for w in words:
            w_up = w.upper()
            # Avoid false positives like VND or USD currency unless specific
            if w_up in self.valid_tickers and w_up not in ['VND', 'USD']:
                ticker = w_up
                break
        
        # If not found via direct symbol, search company names/aliases
        if not ticker:
            for alias in sorted(self.alias_to_ticker.keys(), key=len, reverse=True):
                if alias in q_lower:
                    ticker = self.alias_to_ticker[alias]
                    break

        # Check VND as company if not preceded/followed by currency terms
        if not ticker and 'VND' in self.valid_tickers:
            if 'vndirect' in q_lower or (re.search(r'\bvnd\b', q_lower) and not any(term in q_lower for term in ['bằng vnd', 'đồng (vnd)', 'triệu vnd', 'tỷ vnd', 'nghìn vnd'])):
                ticker = 'VND'

        # Extract years (2010-2029)
        years = re.findall(r'\b(20[12]\d)\b', question)
        
        # Report type classification
        if any(term in q_lower for term in ['công ty mẹ', 'báo cáo riêng', 'bctc riêng', 'báo cáo tài chính riêng']):
            report_type = "separate"
        elif any(term in q_lower for term in ['tổng hợp', 'bctc tổng hợp']):
            report_type = "aggregated"
        else:
            report_type = "consolidated"

        return {
            'ticker': ticker,
            'years': years,
            'report_type': report_type
        }

    def retrieve_tables(self, question, top_k=3):
        meta = self.extract_metadata_from_question(question)
        ticker = meta['ticker']
        years = meta['years']
        report_type = meta['report_type']

        candidates = []
        is_global_fallback = False

        if ticker:
            if years:
                for y in years:
                    key = (ticker.upper(), str(y), report_type.lower())
                    if key in self.catalog_index:
                        candidates.extend(self.catalog_index[key])
                    else:
                        for rt in ['consolidated', 'separate', 'aggregated', 'other']:
                            k_fallback = (ticker.upper(), str(y), rt)
                            if k_fallback in self.catalog_index:
                                candidates.extend(self.catalog_index[k_fallback])
            else:
                for k, items in self.catalog_index.items():
                    if k[0] == ticker.upper():
                        candidates.extend(items)

        if not candidates and ticker:
            for k, items in self.catalog_index.items():
                if k[0] == ticker.upper():
                    candidates.extend(items)

        if not candidates:
            candidates = self.catalog
            is_global_fallback = True

        tokenized_query = clean_tokens(question)

        if is_global_fallback:
            scores = self.global_bm25.get_scores(tokenized_query)
            scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        else:
            corpus = []
            for c in candidates:
                headers_text = " ".join(c['headers'])
                labels_text = " ".join(c['sample_row_labels'])
                doc_text = f"{c['ticker']} {c['year']} {c['report_type']} {headers_text} {labels_text}"
                corpus.append(clean_tokens(doc_text))

            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(tokenized_query)
            scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        top_candidates = [c[0] for c in scored_candidates[:top_k]]
        relevant_docs = list(dict.fromkeys([c['report_id'] for c in top_candidates]))
        relevant_tables = [c['table_ref'] for c in top_candidates]

        return {
            'relevant_docs': relevant_docs,
            'relevant_tables': relevant_tables,
            'top_candidates': top_candidates
        }

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    catalog_path = str(project_root / "processed_data" / "table_catalog.json")
    stock_path = str(project_root / "ViFinQA" / "code_stock.csv")
    
    if os.path.exists(catalog_path):
        retriever = TableRetriever(catalog_path, stock_path)
        sample_q = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"
        res = retriever.retrieve_tables(sample_q)
        print("Question:", sample_q)
        print("Relevant Docs:", res['relevant_docs'])
        print("Relevant Tables:", res['relevant_tables'])
