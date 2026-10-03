import re
import json
import math
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from src.llm_client import LLMClient, T2PANDAS_MODEL


def parse_vietnamese_number(val_str: Any) -> Optional[float]:
    if isinstance(val_str, (int, float, np.number)):
        f_val = float(val_str)
        return f_val if not math.isnan(f_val) else None
    if not isinstance(val_str, str):
        return None

    s = val_str.strip()
    if not s or s in ['-', '—', '–', 'N/A', 'NaN', 'null', 'None']:
        return 0.0

    is_neg = False
    if s.startswith('(') and s.endswith(')'):
        is_neg = True
        s = s[1:-1].strip()

    for unit_word in ['vnđ', 'vnd', 'đồng', 'triệu', 'tỷ', 'nghìn']:
        s = re.sub(rf'\b{unit_word}\b', '', s, flags=re.IGNORECASE).strip()

    s = s.replace('%', '').strip()

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
    except ValueError:
        return None


class Stage2PandasGenerator:
    """
    Stage 2: Takes Canonical Query + Retrieved DataFrames,
    generates Pandas code (via DeepSeek if LLM is active, or Local Heuristic Engine),
    executes code in a sandbox, and returns (pandas_query_string, numeric_answer).
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def generate_and_execute(
        self,
        question: str,
        canonical_query: str,
        target_unit: str,
        dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[str, float]:
        """
        Generates and executes Pandas code.
        Returns (pandas_query, answer).
        """
        if not dfs:
            return ("result = 0.0", 0.0)

        cleaned_code = ""
        answer = None
        success = False

        # 1. Try LLM Generation if enabled
        if self.llm.enabled:
            schema_desc = []
            for var_name, df in dfs.items():
                cols = list(df.columns)
                sample_rows = df.head(3).to_dict(orient="records") if not df.empty else []
                schema_desc.append(
                    f"Variable: {var_name}\n"
                    f"Columns: {cols}\n"
                    f"Sample rows:\n{json.dumps(sample_rows, ensure_ascii=False, indent=2)}\n"
                )
            schema_prompt = "\n".join(schema_desc)

            system_prompt = (
                "You are an expert Python and Pandas Data Engineer for Vietnamese financial reports.\n"
                "Write a short executable Pandas snippet storing the final numeric float answer in variable `result`.\n"
                "Use only the provided DataFrames (df1, df2). Output only Python code inside ```python."
            )
            user_prompt = (
                f"Question: \"{question}\"\n"
                f"Canonical Intent: \"{canonical_query}\"\n"
                f"Target Unit: \"{target_unit}\"\n"
                f"Schemas:\n{schema_prompt}\n\n"
                f"Pandas Query:"
            )

            llm_response = self.llm.generate(T2PANDAS_MODEL, system_prompt, user_prompt)
            cleaned_code = self._clean_code(llm_response)
            if cleaned_code:
                answer, success = self._eval_pandas_code(cleaned_code, dfs)

        # 2. Local Heuristic Engine (fast, deterministic, zero-hallucination)
        if not success or answer is None or math.isnan(answer):
            cleaned_code, answer = self._heuristic_pandas_fallback(question, canonical_query, target_unit, dfs)

        # 3. Unit normalization if necessary
        answer = self._normalize_unit(answer, target_unit)
        if answer is None or math.isnan(answer):
            answer = 0.0

        return (cleaned_code, float(answer))

    def _clean_code(self, code_text: str) -> str:
        if not code_text:
            return ""
        code = re.sub(r'```(?:python)?', '', code_text, flags=re.IGNORECASE)
        code = code.replace('```', '').strip()
        return code.strip()

    def _eval_pandas_code(self, code_str: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[Optional[float], bool]:
        if not code_str:
            return (None, False)

        env = {
            "pd": pd,
            "np": np,
            "parse_vn_num": parse_vietnamese_number,
            "result": None,
            **dfs
        }

        # Try evaluating as single expression
        try:
            val = eval(code_str, env)
            f_val = self._extract_float(val)
            if f_val is not None:
                return (f_val, True)
        except Exception:
            pass

        # Try executing as code block
        try:
            exec_locals = {}
            exec(code_str, env, exec_locals)
            res = exec_locals.get("result", env.get("result"))
            f_val = self._extract_float(res)
            if f_val is not None:
                return (f_val, True)
        except Exception:
            pass

        return (None, False)

    def _extract_float(self, val: Any) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float, np.number)):
            f_val = float(val)
            return f_val if not math.isnan(f_val) else None
        if isinstance(val, (pd.DataFrame, pd.Series, np.ndarray)):
            flat = getattr(val, "values", val).flatten()
            if len(flat) > 0:
                return self._extract_float(flat[0])
            return None
        if isinstance(val, str):
            return parse_vietnamese_number(val)
        return None

    def _heuristic_pandas_fallback(
        self,
        question: str,
        canonical_query: str,
        target_unit: str,
        dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[str, float]:
        """
        Local Heuristic Engine:
        Finds target row by matching financial keywords against table rows,
        parses the numeric value from the latest column.
        """
        q_lower = question.lower()
        metric_keywords = [
            "lãi tiền gửi", "cho vay khách hàng", "chi phí dự phòng", "chi phí phạt",
            "lợi nhuận sau thuế", "doanh thu thuần", "doanh thu", "lợi nhuận gộp",
            "tổng tài sản", "nợ phải trả", "vốn chủ sở hữu", "hàng tồn kho",
            "tiền và các khoản tương đương tiền", "tiền và tương đương tiền",
            "chi phí tài chính", "chi phí bán hàng", "chi phí quản lý",
            "quỹ khen thưởng", "vay ngắn hạn", "nợ ngắn hạn"
        ]
        matched_kw = ""
        for kw in metric_keywords:
            if kw in q_lower:
                matched_kw = kw
                break

        for var_name, df in dfs.items():
            if df.empty:
                continue

            first_col = df.columns[0]
            target_row = None

            # 1. Match full metric keyword
            if matched_kw:
                mask = df[first_col].astype(str).str.lower().str.contains(re.escape(matched_kw), na=False)
                matched_df = df[mask]
                if not matched_df.empty:
                    target_row = matched_df.iloc[0]

            # 2. Fallback: match sub-words
            if target_row is None and matched_kw:
                for sub_w in matched_kw.split():
                    if len(sub_w) >= 3:
                        mask = df[first_col].astype(str).str.lower().str.contains(re.escape(sub_w), na=False)
                        matched_df = df[mask]
                        if not matched_df.empty:
                            target_row = matched_df.iloc[0]
                            break

            # 3. Extract float from last valid numeric cell
            if target_row is not None:
                for col_idx in range(len(target_row) - 1, 0, -1):
                    val = parse_vietnamese_number(target_row.iloc[col_idx])
                    if val is not None and (val != 0.0 or str(target_row.iloc[col_idx]).strip() in ['0', '0.0', '0,0']):
                        query_str = (
                            f"{var_name}[{var_name}.iloc[:, 0].astype(str).str.lower()"
                            f".str.contains('{matched_kw or sub_w}', na=False)].iloc[0, -1]"
                        )
                        return (query_str, float(val))

            # 4. Fallback: check first non-empty numeric cell in DataFrame
            for r_idx in range(min(5, len(df))):
                for c_idx in range(len(df.columns) - 1, 0, -1):
                    val = parse_vietnamese_number(df.iloc[r_idx, c_idx])
                    if val is not None and val != 0.0:
                        return (f"{var_name}.iloc[{r_idx}, {c_idx}]", float(val))

        first_var = list(dfs.keys())[0]
        return (f"{first_var}.iloc[0, -1] if not {first_var}.empty else 0.0", 0.0)

    def _normalize_unit(self, val: float, target_unit: str) -> float:
        if val is None or math.isnan(val) or val == 0:
            return 0.0

        u_lower = target_unit.lower()
        # If value is in raw VND (> 100 million) and question asks for triệu đồng
        if "triệu" in u_lower and abs(val) > 1e8:
            return val / 1e6
        # If value is in raw VND (> 100 billion) and question asks for tỷ đồng
        elif "tỷ" in u_lower and abs(val) > 1e11:
            return val / 1e9

        return float(val)
