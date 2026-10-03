import os
import re
import json
import glob
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

def parse_html_table(table_html):
    """
    Parses HTML table content, expands rowspan/colspan, and converts to a 2D grid of strings.
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return []

    rows = table.find_all('tr')
    if not rows:
        return []

    grid = []
    for r_idx, tr in enumerate(rows):
        cells = tr.find_all(['td', 'th'])
        c_idx = 0
        
        while len(grid) <= r_idx:
            grid.append([])
            
        current_row = grid[r_idx]
        
        for cell in cells:
            while c_idx < len(current_row) and current_row[c_idx] is not None:
                c_idx += 1

            text = cell.get_text(separator=" ", strip=True)
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))

            for r in range(r_idx, r_idx + rowspan):
                while len(grid) <= r:
                    grid.append([])
                target_row = grid[r]
                for c in range(c_idx, c_idx + colspan):
                    while len(target_row) <= c:
                        target_row.append(None)
                    target_row[c] = text
                    
            c_idx += colspan
            
    max_cols = max((len(r) for r in grid), default=0)
    final_grid = []
    for r in grid:
        padded_row = [cell if cell is not None else "" for cell in r]
        if len(padded_row) < max_cols:
            padded_row.extend([""] * (max_cols - len(padded_row)))
        final_grid.append(padded_row)
        
    return final_grid

def clean_and_format_grid(grid):
    """
    Identifies header vs body, flattens headers, propagates section headers,
    and returns a clean pandas DataFrame.
    """
    if not grid or len(grid) < 1:
        return pd.DataFrame()

    headers = grid[0]
    clean_headers = []
    for idx, h in enumerate(headers):
        h_str = h.strip()
        if not h_str:
            h_str = f"Col_{idx+1}"
        base_h = h_str
        count = 1
        while h_str in clean_headers:
            h_str = f"{base_h}_{count}"
            count += 1
        clean_headers.append(h_str)

    data_rows = grid[1:]
    if not data_rows:
        return pd.DataFrame(columns=clean_headers)

    current_section = ""
    processed_rows = []
    for row in data_rows:
        if not any(row):
            continue
        row_label = row[0].strip()
        non_empty_cells = [c for c in row if c.strip()]
        
        if len(non_empty_cells) == 1 and row_label:
            current_section = row_label
            row[0] = f"[{current_section}]"
            
        processed_rows.append(row)

    df = pd.DataFrame(processed_rows, columns=clean_headers)
    return df

def extract_tables_from_report(report_path):
    """
    Reads a single .txt report, extracts all <table> HTML blocks with 1-indexed line numbers.
    """
    report_path = Path(report_path)
    report_id = report_path.parent.name # e.g. AAA_financial_statements_2015_consolidated
    
    with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    tables = []
    in_table = False
    table_lines = []
    start_line = 0

    for idx, line in enumerate(lines, 1):
        if re.search(r'<table\b', line, re.IGNORECASE):
            in_table = True
            start_line = idx
            table_lines = [line]
            if re.search(r'</table>', line, re.IGNORECASE):
                in_table = False
                table_html = "".join(table_lines)
                tables.append({
                    'report_id': report_id,
                    'start_line': start_line,
                    'html': table_html,
                    'file_path': str(report_path)
                })
        elif in_table:
            table_lines.append(line)
            if re.search(r'</table>', line, re.IGNORECASE):
                in_table = False
                table_html = "".join(table_lines)
                tables.append({
                    'report_id': report_id,
                    'start_line': start_line,
                    'html': table_html,
                    'file_path': str(report_path)
                })

    return tables

def process_all_reports(statements_dir, output_csv_dir, output_catalog_file):
    """
    Processes all reports in statements_dir, saves CSVs and catalog metadata.
    """
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_catalog_file), exist_ok=True)
    
    report_files = sorted(glob.glob(os.path.join(statements_dir, "*/*/*/*.txt")))
    print(f"Found {len(report_files)} report files.")

    catalog = []
    total_tables = 0

    for report_file in report_files:
        path_obj = Path(report_file)
        ticker = path_obj.parts[-4] # e.g., AAA
        year = path_obj.parts[-3]   # e.g., 2015
        report_id = path_obj.parts[-2] # e.g., AAA_financial_statements_2015_consolidated
        
        report_type = "other"
        if "consolidated" in report_id.lower():
            report_type = "consolidated"
        elif "separate" in report_id.lower():
            report_type = "separate"
        elif "aggregated" in report_id.lower():
            report_type = "aggregated"

        extracted = extract_tables_from_report(report_file)
        
        for t_idx, t_item in enumerate(extracted, 1):
            grid = parse_html_table(t_item['html'])
            if not grid:
                continue
            df = clean_and_format_grid(grid)
            if df.empty or df.shape[1] == 0:
                continue

            csv_filename = f"{report_id}_table_{t_idx}.csv"
            csv_path = os.path.join(output_csv_dir, csv_filename)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            table_ref = f"{report_id}|{t_item['start_line']}"
            
            row_labels = df.iloc[:, 0].dropna().tolist() if not df.empty else []
            headers = df.columns.tolist()
            
            catalog_entry = {
                'table_ref': table_ref,
                'report_id': report_id,
                'ticker': ticker,
                'year': year,
                'report_type': report_type,
                'start_line': t_item['start_line'],
                'csv_filename': csv_filename,
                'csv_path': f"data/{csv_filename}",
                'num_rows': len(df),
                'num_cols': len(df.columns),
                'headers': headers,
                'sample_row_labels': row_labels[:15]
            }
            catalog.append(catalog_entry)
            total_tables += 1

    with open(output_catalog_file, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"Processed total {total_tables} tables. Saved to {output_catalog_file}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    statements_directory = str(project_root / "ViFinQA" / "financial_statements")
    csv_directory = str(project_root / "processed_data" / "csv")
    catalog_path = str(project_root / "processed_data" / "table_catalog.json")
    
    process_all_reports(statements_directory, csv_directory, catalog_path)
