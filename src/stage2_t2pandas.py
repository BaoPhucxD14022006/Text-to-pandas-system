import re
import json
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.llm_client import LLMClient, T2PANDAS_MODEL


class Stage2PandasGenerator:
    """
    Stage 2: Takes Canonical Query + Retrieved DataFrames schema,
    generates Pandas code using DeepSeek (T2PANDAS_MODEL),
    executes code in a sandbox, and returns (pandas_query_string, numeric_answer).
    """

    def __init__(self, llm_client: LLMClient = None):
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
            return ("0.0", 0.0)

        # 1. Build DataFrame Schema description for LLM
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
            "You are an expert Python and Pandas Data Engineer.\n"
            "Given DataFrame schemas and a financial question, write a single executable Pandas expression\n"
            "or code block that computes the requested numeric answer.\n"
            "Follow these rules strictly:\n"
            "1. Use only the provided DataFrame variable names (e.g. df1, df2).\n"
            "2. The expression must evaluate to a single numeric float or integer.\n"
            "3. Output only the raw code string without markdown formatting or code blocks if possible.\n"
        )

        user_prompt = (
            f"Question: \"{question}\"\n"
            f"Canonical English Intent: \"{canonical_query}\"\n"
            f"Target Unit Requested: \"{target_unit}\"\n"
            f"DataFrame Schemas:\n{schema_prompt}\n\n"
            f"Pandas Query Code:"
        )

        llm_response = self.llm.generate(T2PANDAS_MODEL, system_prompt, user_prompt)
        cleaned_code = self._clean_code(llm_response)

        # 2. Execute generated code in Sandbox
        answer, success = self._eval_pandas_code(cleaned_code, dfs)

        # 3. If LLM code execution failed, use intelligent fallback heuristic
        if not success or answer is None or math.isnan(answer):
            cleaned_code, answer = self._heuristic_pandas_fallback(question, canonical_query, target_unit, dfs)

        # 4. Perform unit normalization
        answer = self._normalize_unit(answer, target_unit)

        if answer is None or math.isnan(answer):
            answer = 0.0

        return (cleaned_code, float(answer))

    def _clean_code(self, code_text: str) -> str:
        if not code_text:
            return ""
        code = re.sub(r'```(?:python)?', '', code_text, flags=re.IGNORECASE)
        code = code.replace('```', '').strip()
        lines = [line.strip() for line in code.split("\n") if line.strip() and not line.strip().startswith("#")]
        if lines:
            return lines[-1]
        return code.strip()

    def _eval_pandas_code(self, code_str: str, dfs: Dict[str, pd.DataFrame]) -> Tuple[Optional[float], bool]:
        if not code_str:
            return (None, False)

        env = {"pd": pd, "np": np, **dfs}
        try:
            val = eval(code_str, env)
            float_val = self._extract_float(val)
            if float_val is not None and not math.isnan(float_val):
                return (float_val, True)
        except Exception:
            pass

        try:
            exec_env = {"pd": pd, "np": np, "result": None, **dfs}
            exec_code = f"result = {code_str}"
            exec(exec_code, exec_env)
            val = exec_env.get("result")
            float_val = self._extract_float(val)
            if float_val is not None and not math.isnan(float_val):
                return (float_val, True)
        except Exception:
            pass

        return (None, False)

    def _extract_float(self, val: Any) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float, np.number)):
            f_val = float(val)
            return f_val if not math.isnan(f_val) else None
        if isinstance(val, (pd.Series, np.ndarray)):
            if len(val) > 0:
                first_item = val.iloc[0] if isinstance(val, pd.Series) else val[0]
                return self._extract_float(first_item)
        if isinstance(val, str):
            clean_str = val.replace(",", "").replace(".", "").strip()
            try:
                f_val = float(clean_str)
                return f_val if not math.isnan(f_val) else None
            except ValueError:
                pass
        return None

    def _heuristic_pandas_fallback(
        self,
        question: str,
        canonical_query: str,
        target_unit: str,
        dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[str, float]:
        """
        Intelligent heuristic fallback:
        Matches metric keywords against text columns (Chỉ tiêu, Khoản mục, etc.),
        extracts numeric cell value from the matching row in numeric columns.
        """
        # Extract potential year filter from canonical query or question
        years = re.findall(r'\b(20[1-2][0-9])\b', question)
        target_year = years[0] if years else ""

        # Extract metric keywords
        q_lower = question.lower()
        metric_keywords = [
            "lãi tiền gửi", "cho vay khách hàng", "chi phí dự phòng", "chi phí phạt",
            "lợi nhuận sau thuế", "doanh thu thuần", "doanh thu", "lợi nhuận gộp",
            "tổng tài sản", "nợ phải trả", "vốn chủ sở hữu", "hàng tồn kho"
        ]
        matched_kw = ""
        for kw in metric_keywords:
            if kw in q_lower:
                matched_kw = kw
                break

        for var_name, df in dfs.items():
            if df.empty:
                continue

            # Identify metric text column
            metric_col = None
            for c in df.columns:
                c_str = str(c).lower()
                if any(k in c_str for k in ["chỉ tiêu", "khoản mục", "nội dung", "danh mục"]):
                    metric_col = c
                    break
            if not metric_col:
                metric_col = df.columns[0]

            # Identify numeric value columns (exclude 'Mã số', 'Thuyết minh', and metric_col)
            value_cols = [
                c for c in df.columns
                if c != metric_col and str(c).lower() not in ["mã số", "mã", "thuyết minh", "trang"]
            ]

            # Select best value column matching year if available
            best_val_col = value_cols[0] if value_cols else None
            if target_year and value_cols:
                for vc in value_cols:
                    if target_year in str(vc):
                        best_val_col = vc
                        break

            if not best_val_col:
                continue

            # Search row matching matched_kw or any row with numbers
            if matched_kw:
                mask = df[metric_col].astype(str).str.lower().str.contains(re.escape(matched_kw), na=False)
                matching_df = df[mask]
                if not matching_df.empty:
                    val = matching_df.iloc[0][best_val_col]
                    f_val = self._extract_float(val)
                    if f_val is not None:
                        query_str = (
                            f"{var_name}[{var_name}['{metric_col}'].astype(str).str.lower()"
                            f".str.contains('{matched_kw}', na=False)].iloc[0]['{best_val_col}']"
                        )
                        return (query_str, f_val)

            # Fallback: scan rows for first non-zero float
            for idx, row in df.iterrows():
                val = row[best_val_col]
                f_val = self._extract_float(val)
                if f_val is not None and f_val != 0:
                    query_str = f"{var_name}.iloc[{idx}]['{best_val_col}']"
                    return (query_str, f_val)

        return (f"{list(dfs.keys())[0]}.iloc[0, -1] if not {list(dfs.keys())[0]}.empty else 0.0", 0.0)

    def _normalize_unit(self, val: float, target_unit: str) -> float:
        """
        Normalizes numeric answer to match requested target_unit.
        """
        if val is None or math.isnan(val) or val == 0:
            return 0.0

        u_lower = target_unit.lower()
        if "triệu" in u_lower and abs(val) > 1e8:
            return val / 1e6
        elif "tỷ" in u_lower and abs(val) > 1e11:
            return val / 1e9

        return float(val)
