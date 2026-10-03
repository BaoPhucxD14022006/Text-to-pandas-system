import os
import re
import json
from pathlib import Path
import pandas as pd

PANDAS_PROMPT_TEMPLATE = """You are an expert Python Pandas data analyst for Vietnamese financial reports.
Your task is to write a short, executable Python Pandas snippet to compute the answer for a financial question.

Question: {question}

Candidate Evidence Tables:
{tables_info}

STRICT CONSTRAINTS & RUNTIME CONTRACT:
1. Do NOT import any libraries. `pd` and `parse_vn_num` are already imported and available in namespace.
2. The evidence DataFrames are pre-loaded in the dictionary `dfs` where key is `<csv_path>` or variable name (`df1`, `df2`, etc.).
3. The first DataFrame is `df1 = dfs['{df1_key}']`.
4. Process Vietnamese numerical strings with `parse_vn_num(val_str)`:
   - Thousands separator is '.' (e.g., '15.230.000' -> 15230000)
   - Decimal separator is ',' (e.g., '12,5%' -> 12.5)
   - Parentheses '(500.000)' means negative -500000
5. Unit Conversions:
   - Check if question asks for 'tỷ đồng' (billion VND) or 'triệu đồng' (million VND) vs numbers in table.
6. The final answer MUST be stored in a variable named `result` as a float or int.

Write ONLY valid Python code block starting with ```python and ending with ```.
"""

def parse_vietnamese_number(val_str):
    """
    Utility function to parse Vietnamese number formatting into float.
    """
    if isinstance(val_str, (int, float)):
        return float(val_str)
    if not isinstance(val_str, str):
        try:
            return float(val_str)
        except:
            return 0.0

    s = val_str.strip()
    if not s or s in ['-', '—', '–', 'N/A', 'NaN', 'null', 'None']:
        return 0.0

    # Negative check: (123.456)
    is_neg = False
    if s.startswith('(') and s.endswith(')'):
        is_neg = True
        s = s[1:-1].strip()

    # Strip currency / unit words if present
    for unit_word in ['vnđ', 'vnd', 'đồng', 'triệu', 'tỷ', 'nghìn']:
        s = re.sub(rf'\b{unit_word}\b', '', s, flags=re.IGNORECASE).strip()

    # Remove % if any
    s = s.replace('%', '').strip()

    # Vietnamese format: '.' as thousand separator, ',' as decimal separator
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s and '.' not in s:
        s = s.replace(',', '.')
    elif '.' in s and ',' not in s:
        # If dot is used as thousand separator (e.g., 15.000.000 or 15.000)
        parts = s.split('.')
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and not is_neg):
            s = s.replace('.', '')
        # else treat as decimal point

    try:
        val = float(s)
        return -val if is_neg else val
    except:
        return 0.0

STOP_WORDS = {
    'trong', 'tính', 'xét', 'tại', 'vào', 'cho', 'của', 'là', 'bao', 'nhiêu', 'năm', 
    'cuối', 'đầu', 'công', 'ty', 'mẹ', 'ctcp', 'tập', 'đoàn', 'ngân', 'hàng', 'tmcp', 
    'giai', 'đoạn', 'mấy', 'triệu', 'tỷ', 'nghìn', 'đồng', 'đến', 'ngày', 'tháng', 
    'nào', 'đạt', 'hãy', 'xác', 'định', 'sau', 'trước', 'với', 'và', 'các', 'những', 
    'được', 'có', 'theo', 'hợp', 'nhất', 'riêng', 'biết', 'thời', 'điểm', 'kỳ', 'cp'
}

def extract_financial_keywords(question):
    """
    Extracts informative financial terms from question text.
    """
    # Remove years and stock tickers in parentheses
    txt = re.sub(r'\b(20[12]\d)\b', ' ', question)
    txt = re.sub(r'\([A-Za-z0-9]+\)', ' ', txt)
    words = [w.lower() for w in re.findall(r'[\w]+', txt) if w.lower() not in STOP_WORDS and not w.isdigit()]
    return words

def generate_heuristic_pandas_query(question, candidate_tables, csv_dir):
    """
    Heuristic rule-based pandas query generator for dry-run and fallback.
    """
    if not candidate_tables:
        return {
            'pandas_query': "result = 0.0",
            'evidence': []
        }

    first_table = candidate_tables[0]
    csv_path = first_table['csv_path']
    full_csv_path = os.path.join(csv_dir, first_table['csv_filename'])

    evidence = [
        {
            "variable": "df1",
            "csv_path": csv_path
        }
    ]

    keywords = extract_financial_keywords(question)
    search_keyword = " ".join(keywords[:3]) if len(keywords) >= 2 else (keywords[0] if keywords else "")

    query_code = f"""df1 = dfs["{csv_path}"]
# Locate target row by matching financial keywords against table rows
target_row = None
keywords = {keywords[:5]}

if not df1.empty:
    first_col = df1.iloc[:, 0].astype(str)
    # 1. Try phrase match
    phrase = "{search_keyword}"
    if phrase:
        matched = df1[first_col.str.contains(phrase, case=False, na=False, regex=False)]
        if not matched.empty:
            target_row = matched.iloc[0]

    # 2. Fallback to individual keyword match
    if target_row is None:
        for kw in keywords:
            if len(kw) >= 3:
                matched = df1[first_col.str.contains(kw, case=False, na=False, regex=False)]
                if not matched.empty:
                    target_row = matched.iloc[0]
                    break

result = 0.0
if target_row is not None:
    # Pick last numeric cell in target row
    for col_idx in range(len(target_row) - 1, 0, -1):
        val = parse_vn_num(str(target_row.iloc[col_idx]))
        if val != 0.0 or str(target_row.iloc[col_idx]).strip() in ['0', '0.0', '0,0']:
            result = float(val)
            break
"""
    return {
        'pandas_query': query_code,
        'evidence': evidence
    }

class LLMPandasGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", device=None):
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Loading LLM {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device != "cuda":
            self.model.to(self.device)
        print("Model loaded successfully.")

    def generate_query(self, question, candidate_tables, csv_dir):
        if not self.model:
            return generate_heuristic_pandas_query(question, candidate_tables, csv_dir)

        tables_info = ""
        evidence = []
        for i, tbl in enumerate(candidate_tables[:2], 1):
            var_name = f"df{i}"
            csv_path = tbl['csv_path']
            evidence.append({"variable": var_name, "csv_path": csv_path})
            
            full_path = os.path.join(csv_dir, tbl['csv_filename'])
            if os.path.exists(full_path):
                try:
                    df_sample = pd.read_csv(full_path).head(10)
                    tables_info += f"\n--- Table {var_name} ({csv_path}) ---\n"
                    tables_info += df_sample.to_string() + "\n"
                except Exception:
                    pass

        prompt = PANDAS_PROMPT_TEMPLATE.format(
            question=question,
            tables_info=tables_info,
            df1_key=evidence[0]['csv_path'] if evidence else ""
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response.strip()

        return {
            'pandas_query': code,
            'evidence': evidence
        }

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    csv_dir = str(project_root / "processed_data" / "csv")
    
    q = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"
    cands = [{'csv_path': 'data/VJC_financial_statements_2018_separate_table_1.csv', 'csv_filename': 'VJC_financial_statements_2018_separate_table_1.csv'}]
    res = generate_heuristic_pandas_query(q, cands, csv_dir)
    print("Generated Query:\n", res['pandas_query'])
