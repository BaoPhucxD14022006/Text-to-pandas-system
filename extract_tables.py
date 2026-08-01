import os
import sys
import re
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def safe_int(val, default=None):
    """Ép kiểu int an toàn, trả về default nếu lỗi (tránh lỗi với ký tự unicode như '①')."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=None):
    """Ép kiểu float an toàn."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class FinancialTableExtractor:
    """
    Trích xuất các bảng dữ liệu từ các file TXT báo cáo tài chính (OCR)
    và lưu dưới dạng JSON theo đúng cấu trúc yêu cầu.
    """

    def __init__(self, stock_csv_path=None):
        self.stock_map = {}
        if stock_csv_path and os.path.exists(stock_csv_path):
            try:
                df_stock = pd.read_csv(stock_csv_path)
                for _, row in df_stock.iterrows():
                    ticker = str(row['Mã CK']).strip()
                    cname = str(row['Tên công ty']).strip()
                    self.stock_map[ticker] = cname
            except Exception as e:
                print(f"Cảnh báo: Không thể đọc file code_stock.csv ({e})")

    def parse_value(self, val_str, col_name):
        """Chuyển đổi chuỗi số (VND, ngoặc đơn, gạch ngang) sang kiểu int/float hoặc giữ nguyên chuỗi."""
        if not isinstance(val_str, str):
            return val_str
        
        s = val_str.strip()
        
        # Giữ nguyên kiểu chuỗi cho các cột mã/tên/ghi chú không phải số tiền
        if col_name in ['Mã số', 'Chỉ tiêu', 'Thuyết minh']:
            return s
            
        if not s or s in ['-', '—', '- ', '..', '...']:
            return 0
            
        is_neg = False
        if s.startswith('(') and s.endswith(')'):
            is_neg = True
            s = s[1:-1].strip()
        elif s.startswith('-'):
            is_neg = True
            s = s[1:].strip()
            
        # Thử ép kiểu số nguyên (dùng isdecimal thay vì isdigit để tránh ký tự unicode như '①')
        clean_digits = s.replace('.', '').replace(',', '')
        if clean_digits.isdecimal():
            val = safe_int(clean_digits)
            if val is not None:
                return -val if is_neg else val
            
        # Thử ép kiểu số thực (ví dụ: 12.345,67 -> 12345.67)
        clean_float = s.replace('.', '').replace(',', '.')
        val_f = safe_float(clean_float)
        if val_f is not None:
            return -val_f if is_neg else val_f

        return s

    def normalize_headers(self, header_cells):
        """Chuẩn hóa tên cột thành các tên tiêu chuẩn trong Báo cáo tài chính."""
        mapped = []
        for h in header_cells:
            h_clean = h.strip()
            h_upper = h_clean.upper()
            
            if 'MÃ SỐ' in h_upper or 'MÃ' in h_upper:
                mapped.append('Mã số')
            elif any(k in h_upper for k in ['TÀI SẢN', 'NGUỒN VỐN', 'CHỈ TIÊU', 'KHOẢN MỤC', 'DANH MỤC', 'NỘI DUNG']):
                mapped.append('Chỉ tiêu')
            elif 'THUYẾT MINH' in h_upper or 'TM' in h_upper:
                mapped.append('Thuyết minh')
            elif any(k in h_upper for k in ['31/12', 'NĂM NAY', 'CUỐI KỲ', 'CUỐI NĂM', 'KỲ NÀY']):
                mapped.append('Số cuối năm')
            elif any(k in h_upper for k in ['01/01', '1/1/', '1/1 ', 'NĂM TRƯỚC', 'ĐẦU KỲ', 'ĐẦU NĂM', 'KỲ TRƯỚC']):
                mapped.append('Số đầu năm')
            else:
                # Nếu là cột ngày năm cụ thể (VD: "Năm 2015", "Năm 2014")
                if re.search(r'NĂM\s*\d{4}', h_upper):
                    if len(mapped) >= 3 and 'Số cuối năm' not in mapped:
                        mapped.append('Số cuối năm')
                    elif 'Số đầu năm' not in mapped:
                        mapped.append('Số đầu năm')
                    else:
                        mapped.append(h_clean)
                else:
                    mapped.append(h_clean if h_clean else 'Chỉ tiêu')
        return mapped

    def clean_report_type(self, title_str):
        """Làm sạch tên báo cáo / tiêu đề bảng."""
        if not title_str:
            return 'Báo cáo tài chính'
        t = title_str.strip()
        t_upper = t.upper()
        
        if 'BẢNG CÂN ĐỐI KẾ TOÁN' in t_upper:
            return 'Bảng cân đối kế toán'
        if 'KẾT QUẢ HOẠT ĐỘNG KINH DOANH' in t_upper or 'KẾT QUẢ KINH DOANH' in t_upper:
            return 'Báo cáo kết quả hoạt động kinh doanh'
        if 'LƯU CHUYỂN TIỀN TỆ' in t_upper:
            return 'Báo cáo lưu chuyển tiền tệ'
        if 'THUYẾT MINH' in t_upper:
            return 'Thuyết minh báo cáo tài chính'
            
        t = re.sub(r'\s*\(TIẾP THEO\)', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*HỢP NHẤT', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s*RIÊNG LẺ', '', t, flags=re.IGNORECASE)
        return t.strip()

    def extract_from_file(self, file_path, merge_pages=True):
        """Trích xuất danh sách các table JSON từ 1 file txt OCR."""
        file_path_str = str(file_path).replace('\\', '/')
        parts = file_path_str.split('/')
        
        ticker = ''
        year = None
        for p in parts:
            if p in self.stock_map:
                ticker = p
            p_val = safe_int(p)
            if p.isdecimal() and len(p) == 4 and p_val and 2000 <= p_val <= 2030:
                year = p_val
                
        company = self.stock_map.get(ticker, ticker if ticker else 'Unknown Company')
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            print(f"Lỗi đọc file {file_path}: {e}")
            return []
            
        # Nếu chưa tìm thấy company trong stock_map, thử tìm trong 1000 ký tự đầu file txt
        if company == ticker or company == 'Unknown Company':
            m = re.search(r'(CÔNG TY\s+[^\n]+)', content[:1000], re.IGNORECASE)
            if m:
                company = m.group(1).strip()

        pages_raw = re.split(r'=====\s*PAGE\s*(\d+)\s*=====', content)
        raw_tables = []
        
        if len(pages_raw) == 1:
            pages_list = [(1, content)]
        else:
            pages_list = []
            for i in range(1, len(pages_raw), 2):
                p_num = safe_int(pages_raw[i], default=i//2 + 1)
                pages_list.append((p_num, pages_raw[i+1]))

        for ocr_p, p_text in pages_list:
            # Phát hiện số trang in ở cuối trang (chỉ chấp nhận các số nguyên isdecimal)
            printed_p = str(ocr_p)
            lines = [l.strip() for l in p_text.split('\n') if l.strip()]
            for line in reversed(lines[-5:]):
                if line.isdecimal():
                    p_val = safe_int(line)
                    if p_val is not None and 0 < p_val < 500:
                        printed_p = str(p_val)
                        break

            tables_html = re.findall(r'<table>.*?</table>', p_text, re.DOTALL)
            for t_html in tables_html:
                idx = p_text.find(t_html)
                ctx = p_text[:idx].strip().split('\n')
                
                raw_title = 'Báo cáo tài chính'
                for line in reversed(ctx[-10:]):
                    l_str = line.strip()
                    if any(k in l_str.upper() for k in [
                        'BẢNG CÂN ĐỐI KẾ TOÁN', 'BÁO CÁO KẾT QUẢ', 
                        'BÁO CÁO LƯU CHUYỂN', 'THUYẾT MINH', 'BÁO CÁO'
                    ]):
                        raw_title = l_str
                        break
                        
                report_type = self.clean_report_type(raw_title)
                
                # Parse HTML table matrix bằng BeautifulSoup
                soup = BeautifulSoup(t_html, 'html.parser')
                tr_elements = soup.find_all('tr')
                matrix = []
                for tr in tr_elements:
                    cells = [c.get_text(strip=True) for c in tr.find_all(['td', 'th'])]
                    if cells and any(cells):
                        matrix.append(cells)
                        
                if not matrix:
                    continue
                    
                headers = self.normalize_headers(matrix[0])
                
                rows_data = []
                for r_cells in matrix[1:]:
                    if not any(r_cells):
                        continue
                    row_dict = {}
                    for h, v in zip(headers, r_cells):
                        row_dict[h] = self.parse_value(v, h)
                    
                    ordered_row = {}
                    for k in ['Mã số', 'Chỉ tiêu', 'Thuyết minh', 'Số cuối năm', 'Số đầu năm']:
                        if k in row_dict:
                            ordered_row[k] = row_dict[k]
                    for k, v in row_dict.items():
                        if k not in ordered_row:
                            ordered_row[k] = v
                            
                    rows_data.append(ordered_row)
                    
                if rows_data:
                    raw_tables.append({
                        "company": company,
                        "year": year,
                        "report_type": report_type,
                        "page": printed_p,
                        "path_file": file_path_str,
                        "table_data": rows_data
                    })

        if merge_pages:
            return self.merge_consecutive_tables(raw_tables)
        return raw_tables

    def merge_consecutive_tables(self, table_items):
        """Gộp các bảng cùng loại nối tiếp nhau qua nhiều trang (ví dụ Bảng cân đối kế toán trang 6-7)."""
        if not table_items:
            return []
            
        merged = []
        i = 0
        while i < len(table_items):
            curr = table_items[i]
            if i + 1 < len(table_items):
                nxt = table_items[i+1]
                same_type = (curr['report_type'] == nxt['report_type'])
                same_company = (curr['company'] == nxt['company'])
                same_year = (curr['year'] == nxt['year'])
                
                curr_p = safe_int(str(curr['page']).split('-')[0])
                nxt_p = safe_int(str(nxt['page']).split('-')[0])
                
                is_consecutive = False
                if curr_p is not None and nxt_p is not None:
                    is_consecutive = (nxt_p == curr_p + 1 or nxt_p == curr_p)
                    
                if same_company and same_year and same_type and is_consecutive:
                    page_range = f"{curr['page']}-{nxt['page']}" if curr['page'] != nxt['page'] else str(curr['page'])
                    merged_item = {
                        "company": curr['company'],
                        "year": curr['year'],
                        "report_type": curr['report_type'],
                        "page": page_range,
                        "path_file": curr.get("path_file", ""),
                        "table_data": curr['table_data'] + nxt['table_data']
                    }
                    merged.append(merged_item)
                    i += 2
                    continue
                    
            merged.append(curr)
            i += 1
            
        return merged

def main():
    parser = argparse.ArgumentParser(description="Trích xuất Table từ OCR Báo cáo tài chính sang JSON.")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn tới file txt hoặc thư mục chứa các file txt báo cáo tài chính.")
    parser.add_argument("--output", "-o", required=True, help="Đường dẫn file JSON đầu ra.")
    parser.add_argument("--stock-csv", default="d:/AI_Guru_contest/ViFinQA/code_stock.csv", help="Đường dẫn file code_stock.csv ánh xạ mã CK -> tên công ty.")
    parser.add_argument("--no-merge", action="store_true", help="Tắt tính năng gộp các bảng nối tiếp nhau trên nhiều trang.")
    
    args = parser.parse_args()
    
    extractor = FinancialTableExtractor(stock_csv_path=args.stock_csv)
    
    input_path = Path(args.input)
    all_tables = []
    
    if input_path.is_file():
        print(f"Đang xử lý file: {input_path}")
        tables = extractor.extract_from_file(input_path, merge_pages=not args.no_merge)
        all_tables.extend(tables)
    elif input_path.is_dir():
        txt_files = list(input_path.glob("**/*.txt"))
        print(f"Tìm thấy {len(txt_files)} file txt trong {input_path}")
        for idx, tf in enumerate(txt_files, 1):
            if idx % 100 == 0 or idx == len(txt_files):
                print(f"Đang xử lý [{idx}/{len(txt_files)}] {tf.name}...")
            try:
                tables = extractor.extract_from_file(tf, merge_pages=not args.no_merge)
                all_tables.extend(tables)
            except Exception as e:
                print(f"Cảnh báo: Bỏ qua file lỗi {tf.name}: {e}")
    else:
        print(f"Lỗi: Không tìm thấy đường dẫn {input_path}")
        sys.exit(1)
        
    print(f"Tổng cộng đã trích xuất {len(all_tables)} bảng.")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)
        
    print(f"Đã lưu kết quả thành công vào: {output_path}")

if __name__ == "__main__":
    main()
