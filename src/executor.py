import os
import re
import math
from pathlib import Path
import pandas as pd

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

    is_neg = False
    if s.startswith('(') and s.endswith(')'):
        is_neg = True
        s = s[1:-1].strip()

    # Strip currency / unit words if present
    for unit_word in ['vnđ', 'vnd', 'đồng', 'triệu', 'tỷ', 'nghìn']:
        s = re.sub(rf'\b{unit_word}\b', '', s, flags=re.IGNORECASE).strip()

    s = s.replace('%', '').strip()

    # Vietnamese format: '.' as thousand separator, ',' as decimal separator
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s and '.' not in s:
        s = s.replace(',', '.')
    elif '.' in s and ',' not in s:
        parts = s.split('.')
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and not is_neg):
            s = s.replace('.', '')

    try:
        val = float(s)
        return -val if is_neg else val
    except:
        return 0.0

def execute_pandas_query(query_code, evidence_list, csv_dir):
    """
    Safely executes pandas_query string against pre-loaded DataFrames.
    Returns calculated float answer and success status.
    """
    dfs = {}
    for item in evidence_list:
        csv_path_rel = item['csv_path']
        csv_name = os.path.basename(csv_path_rel)
        full_path = os.path.join(csv_dir, csv_name)
        
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path, dtype=str)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        dfs[csv_path_rel] = df
        dfs[item['variable']] = df

    # Prepare execution namespace
    exec_globals = {
        'pd': pd,
        'dfs': dfs,
        'parse_vn_num': parse_vietnamese_number,
        'parse_vietnamese_number': parse_vietnamese_number
    }
    
    # Assign df1, df2 directly to local variables if defined in evidence
    for item in evidence_list:
        exec_globals[item['variable']] = dfs[item['variable']]

    exec_locals = {}

    try:
        exec(query_code, exec_globals, exec_locals)
        res = exec_locals.get('result', exec_globals.get('result', None))
        
        if res is not None:
            # Handle pandas DataFrame / Series / numpy array safely
            if hasattr(res, 'values'):
                vals = res.values
                if hasattr(vals, 'flatten'):
                    flat = vals.flatten()
                    res = flat[0] if len(flat) > 0 else 0.0
                elif len(vals) > 0:
                    res = vals[0]
                else:
                    res = 0.0

            if hasattr(res, 'item'):
                try:
                    res = res.item()
                except Exception:
                    pass
                
            # If string returned, parse it
            if isinstance(res, str):
                res = parse_vietnamese_number(res)

            res_float = float(res)
            if math.isnan(res_float) or math.isinf(res_float):
                return 0.0, False
            return res_float, True
        else:
            return 0.0, False
    except Exception as e:
        # Execution error
        return 0.0, False

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    csv_dir = str(project_root / "processed_data" / "csv")
    
    sample_code = """
df1 = dfs['data/sample.csv']
result = parse_vn_num('15.230.000,5')
"""
    ans, ok = execute_pandas_query(sample_code, [{'variable': 'df1', 'csv_path': 'data/sample.csv'}], csv_dir)
    print("Test Execution Output:", ans, "Success:", ok)
