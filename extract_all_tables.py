"""
extract_all_tables.py
=====================
Trích xuất TẤT CẢ các bảng HTML (<table>...</table>) từ các file .txt
báo cáo tài chính OCR trong thư mục ViFinQA, convert sang JSON có cấu trúc.

Xử lý đặc biệt:
  - Header "31/12/2025VND" → tách thành ngày "31/12/2025" và đơn vị "VND"
  - Giá trị "(12.345)" → -12345
  - Giá trị "1.234.567" → 1234567 (số VND)
  - Giá trị "-" hoặc rỗng → None
  - Giữ nguyên metadata: tiêu đề bảng, trang, đơn vị, thời gian
"""

import os
import re
import sys
import glob
import json
import argparse
from bs4 import BeautifulSoup

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────────────────────────
# 1. Phân tách header kiểu "31/12/2025VND" hoặc "Năm 2015VND"
# ──────────────────────────────────────────────────────────────────
CURRENCY_UNITS = ['VND', 'USD', 'EUR', 'JPY', 'GBP', 'CNY', 'KRW', 'TWD', 'SGD', 'THB', 'LAK']

def split_header_unit(header_text):
    """
    Tách header chứa đơn vị tiền tệ dính liền.
    Ví dụ:
      "31/12/2025VND"  -> ("31/12/2025", "VND")
      "Năm 2015VND"    -> ("Năm 2015", "VND")
      "Dưới 1 năm VND" -> ("Dưới 1 năm", "VND")
      "Tổng VND"       -> ("Tổng", "VND")
      "TÀI SẢN"        -> ("TÀI SẢN", None)
    """
    text = header_text.strip()
    for unit in CURRENCY_UNITS:
        # Kiểm tra kết thúc bằng đơn vị tiền (dính liền hoặc cách dấu cách)
        if text.upper().endswith(unit):
            prefix = text[:-len(unit)].rstrip()
            if prefix:  # Chỉ tách nếu phần trước không rỗng
                return prefix, unit
            else:
                return text, None
    return text, None


# ──────────────────────────────────────────────────────────────────
# 2. Parse giá trị ô trong bảng tài chính
# ──────────────────────────────────────────────────────────────────
def parse_cell_value(raw_value):
    """
    Chuyển đổi giá trị ô từ chuỗi OCR sang giá trị Python.
    
    Rules:
      - "" hoặc "-" hoặc "—"  → None
      - "(1.234.567)"         → -1234567  (số âm)
      - "1.234.567"           → 1234567   (số VND dùng dấu . phân nhóm)
      - "12,5"                → 12.5      (số thập phân VN)
      - Chuỗi text            → giữ nguyên string
    """
    s = raw_value.strip()
    
    # Rỗng hoặc gạch ngang
    if not s or s in ['-', '—', '- ', '..', '...', '–']:
        return None
    
    # Kiểm tra số âm trong ngoặc đơn: (12.345.678)
    is_negative = False
    if s.startswith('(') and s.endswith(')'):
        is_negative = True
        s = s[1:-1].strip()
    
    # ── Phân biệt dấu chấm thập phân vs phân nhóm nghìn VND ──
    # VND format: "1.234.567.890" (nhóm 3 chữ số sau mỗi dấu chấm)
    # Thập phân:  "5.1", "5.12", "3.5" (1-2 chữ số sau dấu chấm cuối)
    
    # Kiểm tra nếu là số nguyên thuần (không có dấu chấm)
    if re.match(r'^\d+$', s):
        val = int(s)
        return -val if is_negative else val
    
    # Kiểm tra format VND phân nhóm nghìn: N.NNN.NNN hoặc N.NNN
    # Mỗi nhóm sau dấu chấm đầu tiên phải có đúng 3 chữ số
    vnd_pattern = re.match(r'^(\d{1,3})((?:\.\d{3})+)$', s)
    if vnd_pattern:
        clean = s.replace('.', '')
        val = int(clean)
        return -val if is_negative else val
    
    # Kiểm tra format số thập phân với dấu phẩy: "12,5" → 12.5
    comma_decimal = re.match(r'^(\d+),(\d+)$', s)
    if comma_decimal:
        val = float(s.replace(',', '.'))
        return -val if is_negative else val
    
    # Kiểm tra format VND + thập phân: "1.234.567,89"
    vnd_decimal = re.match(r'^(\d{1,3}(?:\.\d{3})*),(\d+)$', s)
    if vnd_decimal:
        integer_part = vnd_decimal.group(0).split(',')[0].replace('.', '')
        decimal_part = vnd_decimal.group(2)
        val = float(f"{integer_part}.{decimal_part}")
        return -val if is_negative else val
    
    # Nếu không match các pattern số ở trên → giữ nguyên chuỗi
    if is_negative:
        return f"({s})"
    return raw_value.strip()


# ──────────────────────────────────────────────────────────────────
# 3. Parse HTML table thành JSON có cấu trúc
# ──────────────────────────────────────────────────────────────────
def parse_html_table(table_html):
    """
    Parse 1 đoạn HTML <table>...</table> thành dạng JSON có cấu trúc.
    
    Returns:
        {
            "headers": [
                {"name": "TÀI SẢN", "unit": null},
                {"name": "31/12/2015", "unit": "VND"},
                ...
            ],
            "rows": [
                {"TÀI SẢN": "A. TÀI SẢN NGẮN HẠN", "Mã số": "100", ...},
                ...
            ]
        }
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return None
    
    all_rows = table.find_all('tr')
    if not all_rows:
        return None
    
    # ── Xác định header rows ──
    # Tìm các dòng header (có thể có rowspan/colspan)
    # Chiến lược: dòng đầu tiên luôn là header.
    # Nếu có rowspan > 1 ở dòng đầu → dòng tiếp cũng là header.
    header_rows = []
    data_rows = []
    
    # Kiểm tra rowspan ở dòng đầu
    first_row_cells = all_rows[0].find_all(['td', 'th'])
    max_rowspan = 1
    for cell in first_row_cells:
        rs = int(cell.get('rowspan', 1))
        if rs > max_rowspan:
            max_rowspan = rs
    
    num_header_rows = max_rowspan
    header_rows = all_rows[:num_header_rows]
    data_rows = all_rows[num_header_rows:]
    
    # ── Xây dựng danh sách header names ──
    if num_header_rows == 1:
        # Đơn giản: 1 dòng header
        headers_raw = []
        for cell in first_row_cells:
            colspan = int(cell.get('colspan', 1))
            text = cell.get_text(strip=True)
            for _ in range(colspan):
                headers_raw.append(text)
        
        # Tách unit ra khỏi header
        headers = []
        for h in headers_raw:
            name, unit = split_header_unit(h)
            headers.append({"name": name, "unit": unit})
    
    elif num_header_rows == 2:
        # Hai dòng header (có rowspan/colspan)
        # Xây dựng grid header
        row1_cells = header_rows[0].find_all(['td', 'th'])
        row2_cells = header_rows[1].find_all(['td', 'th'])
        
        # Đếm số cột tối đa
        max_cols = 0
        for row in all_rows:
            cols = 0
            for cell in row.find_all(['td', 'th']):
                cols += int(cell.get('colspan', 1))
            max_cols = max(max_cols, cols)
        
        # Tạo grid 2 x max_cols
        grid = [['' for _ in range(max_cols)] for _ in range(2)]
        
        # Fill row 1
        col_idx = 0
        for cell in row1_cells:
            rs = int(cell.get('rowspan', 1))
            cs = int(cell.get('colspan', 1))
            text = cell.get_text(strip=True)
            while col_idx < max_cols and grid[0][col_idx] != '':
                col_idx += 1
            for r in range(rs):
                for c in range(cs):
                    if col_idx + c < max_cols and r < 2:
                        grid[r][col_idx + c] = text
            col_idx += cs
        
        # Fill row 2
        col_idx = 0
        for cell in row2_cells:
            cs = int(cell.get('colspan', 1))
            text = cell.get_text(strip=True)
            while col_idx < max_cols and grid[1][col_idx] != '':
                col_idx += 1
            for c in range(cs):
                if col_idx + c < max_cols:
                    grid[1][col_idx + c] = text
            col_idx += cs
        
        # Merge 2 dòng header
        headers = []
        for c in range(max_cols):
            top = grid[0][c].strip()
            bottom = grid[1][c].strip()
            if top == bottom or not bottom:
                combined = top
            elif not top:
                combined = bottom
            else:
                combined = f"{top} - {bottom}"
            name, unit = split_header_unit(combined)
            headers.append({"name": name, "unit": unit})
    else:
        # Fallback: chỉ lấy dòng đầu
        headers_raw = []
        for cell in first_row_cells:
            text = cell.get_text(strip=True)
            headers_raw.append(text)
        headers = []
        for h in headers_raw:
            name, unit = split_header_unit(h)
            headers.append({"name": name, "unit": unit})
        data_rows = all_rows[1:]
    
    num_cols = len(headers)
    
    # ── Parse data rows ──
    rows_data = []
    for row in data_rows:
        cells = row.find_all(['td', 'th'])
        
        # Kiểm tra colspan (dòng section header spanning toàn bộ)
        if len(cells) == 1:
            cs = int(cells[0].get('colspan', 1))
            if cs >= num_cols - 1:
                # Section header row
                section_text = cells[0].get_text(strip=True)
                row_dict = {"_section_header": section_text}
                rows_data.append(row_dict)
                continue
        
        # Dòng data bình thường
        # Expand colspan
        expanded_values = []
        for cell in cells:
            cs = int(cell.get('colspan', 1))
            text = cell.get_text(strip=True)
            expanded_values.append(text)
            for _ in range(cs - 1):
                expanded_values.append('')
        
        row_dict = {}
        for idx, val in enumerate(expanded_values):
            if idx < num_cols:
                header_name = headers[idx]["name"]
                # Parse giá trị ô
                parsed = parse_cell_value(val)
                row_dict[header_name] = parsed
        
        if row_dict:
            rows_data.append(row_dict)
    
    return {
        "headers": headers,
        "rows": rows_data
    }


# ──────────────────────────────────────────────────────────────────
# 4. Tìm metadata/tiêu đề gần nhất phía trước mỗi bảng
# ──────────────────────────────────────────────────────────────────
def find_table_metadata(lines, table_line_idx):
    """
    Quét lên trên từ vị trí bảng để tìm:
      - Tiêu đề bảng (dòng text in hoa / dòng tiêu đề)
      - Thời gian (Tại ngày..., Cho năm...)
      - Đơn vị (Đơn vị: VND)
      - Trang (===== PAGE N =====)
      - Mẫu biểu (MẪU B 01-DN/HN)
    """
    metadata = {
        "tiêu_đề": None,
        "thời_gian": None,
        "đơn_vị": None,
        "trang": None,
        "mẫu_biểu": None,
        "ngữ_cảnh": None,  # Dòng context ngay trước bảng
    }
    
    # Quét tối đa 20 dòng lên trên
    context_lines = []
    start = max(0, table_line_idx - 20)
    
    for i in range(table_line_idx - 1, start - 1, -1):
        line = lines[i].strip()
        if not line:
            continue
        
        # Tìm trang
        page_match = re.match(r'={5}\s*PAGE\s+(\d+)\s*={5}', line)
        if page_match:
            metadata["trang"] = int(page_match.group(1))
            break  # Đã tìm đến đầu trang, dừng lại
        
        # Tìm đơn vị
        unit_match = re.match(r'(?:Đơn vị|Dom vi|Đ\w+ v\w+)\s*:\s*(.+)', line, re.IGNORECASE)
        if unit_match:
            metadata["đơn_vị"] = unit_match.group(1).strip()
            continue
        
        # Tìm mẫu biểu
        mau_match = re.match(r'(MẪU|MÃU|MẦU)\s+(SỐ|SÓ|SO)?\s*(B\s*\d+.*)', line, re.IGNORECASE)
        if mau_match:
            metadata["mẫu_biểu"] = line.strip()
            continue
        
        # Tìm thời gian
        time_patterns = [
            r'(Tại ngày\s+.+)',
            r'(Cho năm tài chính kết thúc.+)',
            r'(Cho năm tài chính kết thúc)',
            r'(ngày\s+\d+\s+tháng\s+\d+\s+năm\s+\d+)',
        ]
        for tp in time_patterns:
            tmatch = re.match(tp, line, re.IGNORECASE)
            if tmatch:
                if metadata["thời_gian"]:
                    metadata["thời_gian"] = line.strip() + " " + metadata["thời_gian"]
                else:
                    metadata["thời_gian"] = line.strip()
                break
        
        # Thu thập context lines (để tìm tiêu đề)
        context_lines.insert(0, line)
    
    # Xác định tiêu đề: ưu tiên dòng in hoa hoặc dòng có keyword đặc biệt
    title_keywords = [
        'BẢNG CÂN ĐỐI', 'BÁO CÁO KẾT QUẢ', 'BÁO CÁO LƯU CHUYỂN',
        'BẢO CẢO LƯU CHUYÊN', 'BÁO CÁO KÊT QUẢ',
        'THUYẾT MINH', 'THUYÊT MINH', 'MỤC LỤC',
        'BẢNG CÂN ĐỒI', 'BẢNG CẦN ĐỐI', 'BẢNG CÂN ĐỐI'
    ]
    
    for cl in context_lines:
        cl_upper = cl.upper()
        for kw in title_keywords:
            if kw in cl_upper:
                metadata["tiêu_đề"] = cl
                break
        if metadata["tiêu_đề"]:
            break
    
    # Nếu chưa tìm thấy tiêu đề bằng keyword, tìm dòng context ngay trước bảng
    if not metadata["tiêu_đề"]:
        # Lấy dòng non-empty cuối cùng trước bảng (bỏ qua đơn vị, mẫu, thời gian)
        for cl in reversed(context_lines):
            cl_stripped = cl.strip()
            if cl_stripped and not re.match(r'(Đơn vị|Dom vi|MẪU|MÃU|MẦU|Tại ngày|Cho năm|ngày \d+|===)', cl_stripped, re.IGNORECASE):
                if len(cl_stripped) > 5:  # Bỏ qua dòng quá ngắn (số trang, ký hiệu)
                    metadata["ngữ_cảnh"] = cl_stripped
                    break
    
    return metadata


# ──────────────────────────────────────────────────────────────────
# 5. Trích xuất TẤT CẢ bảng từ 1 file .txt
# ──────────────────────────────────────────────────────────────────
def extract_all_tables_from_file(file_path):
    """
    Đọc file .txt, tìm tất cả <table>...</table> và trả về danh sách JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Lỗi đọc file {file_path}: {e}")
        return []
    
    content = ''.join(lines)
    results = []
    
    # Tìm tất cả <table>...</table> trong file
    table_pattern = re.compile(r'<table>.*?</table>', re.DOTALL | re.IGNORECASE)
    
    for match in table_pattern.finditer(content):
        table_html = match.group()
        
        # Xác định dòng chứa <table>
        char_pos = match.start()
        line_idx = content[:char_pos].count('\n')
        
        # Parse HTML table → JSON
        parsed = parse_html_table(table_html)
        if not parsed or not parsed["rows"]:
            continue
        
        # Tìm metadata/tiêu đề
        metadata = find_table_metadata(lines, line_idx)
        
        # Build kết quả
        table_entry = {
            "file_path": os.path.normpath(file_path),
            "trang": metadata["trang"],
            "tiêu_đề": metadata["tiêu_đề"],
            "thời_gian": metadata["thời_gian"],
            "đơn_vị": metadata["đơn_vị"],
            "mẫu_biểu": metadata["mẫu_biểu"],
            "ngữ_cảnh": metadata["ngữ_cảnh"],
            "số_dòng_dữ_liệu": len(parsed["rows"]),
            "headers": parsed["headers"],
            "rows": parsed["rows"],
            "html_gốc": table_html,
        }
        
        results.append(table_entry)
    
    return results


# ──────────────────────────────────────────────────────────────────
# 6. Xử lý toàn bộ thư mục
# ──────────────────────────────────────────────────────────────────
def extract_all_tables(target_dir="ViFinQA", output_file="all_tables_structured.json", sample_only=False):
    """
    Quét toàn bộ .txt trong target_dir, trích xuất tất cả bảng.
    """
    txt_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.txt"), recursive=True))
    print(f"Tìm thấy {len(txt_files)} file .txt trong {target_dir}")
    
    all_tables = []
    file_count = 0
    
    for file_path in txt_files:
        tables = extract_all_tables_from_file(file_path)
        if tables:
            all_tables.extend(tables)
            file_count += 1
        
        if sample_only and all_tables:
            break  # Chỉ xử lý 1 file cho sample
    
    print(f"\n=> Đã xử lý {file_count} file, tổng số bảng: {len(all_tables)}")
    
    return all_tables


# ──────────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Trích xuất tất cả bảng từ file .txt báo cáo tài chính (OCR) sang JSON có cấu trúc."
    )
    parser.add_argument("--dir", type=str, default="ViFinQA",
                        help="Thư mục gốc chứa file .txt (Mặc định: ViFinQA)")
    parser.add_argument("--output", type=str, default="all_tables_structured.json",
                        help="File JSON đầu ra (Mặc định: all_tables_structured.json)")
    parser.add_argument("--sample", action="store_true",
                        help="Chỉ xử lý 1 file đầu tiên để xem mẫu")
    parser.add_argument("--file", type=str, default=None,
                        help="Xử lý 1 file cụ thể thay vì toàn bộ thư mục")
    parser.add_argument("--no-html", action="store_true",
                        help="Không giữ html_gốc trong output (tiết kiệm dung lượng)")
    
    args = parser.parse_args()
    
    if args.file:
        # Xử lý 1 file cụ thể
        print(f"Đang xử lý file: {args.file}")
        tables = extract_all_tables_from_file(args.file)
        print(f"=> Tìm thấy {len(tables)} bảng")
    else:
        tables = extract_all_tables(
            target_dir=args.dir,
            output_file=args.output,
            sample_only=args.sample
        )
    
    if not tables:
        print("Không tìm thấy bảng nào.")
        return
    
    # Loại bỏ html_gốc nếu --no-html
    if args.no_html:
        for t in tables:
            t.pop("html_gốc", None)
    
    # In sample
    print("\n" + "=" * 70)
    print("MẪU DỮ LIỆU (3 bảng đầu tiên)")
    print("=" * 70)
    
    for i, table in enumerate(tables[:3]):
        sample = {k: v for k, v in table.items() if k != 'html_gốc'}
        # Giới hạn rows hiển thị
        if len(sample.get("rows", [])) > 5:
            sample["rows"] = sample["rows"][:5] + [{"...": f"(còn {len(table['rows']) - 5} dòng nữa)"}]
        
        print(f"\n--- Bảng {i + 1} ---")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 70)
    
    # Lưu file
    if not args.sample or args.file:
        # Loại bỏ html_gốc khi lưu (quá lớn)
        save_data = []
        for t in tables:
            save_entry = {k: v for k, v in t.items() if k != 'html_gốc'}
            save_data.append(save_entry)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Đã lưu {len(save_data)} bảng vào: {args.output}")


if __name__ == "__main__":
    main()
