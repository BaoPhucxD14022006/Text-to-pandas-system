import os
import json
import math
import zipfile
from typing import Dict, Any, List


class SubmissionBuilder:
    """
    Formats final system outputs into submission.json and packages submission.zip
    according to strict submission requirements.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def build_submission(self, predictions: List[Dict[str, Any]], zip_filepath: str = "submission.zip") -> str:
        """
        Writes submission.json and compresses output_dir into zip_filepath.
        """
        json_path = os.path.join(self.output_dir, "submission.json")

        formatted_predictions = []
        for item in predictions:
            raw_ans = item.get("answer", 0.0)
            try:
                ans_float = float(raw_ans)
                if math.isnan(ans_float) or math.isinf(ans_float):
                    ans_float = 0.0
            except (ValueError, TypeError):
                ans_float = 0.0

            formatted_item = {
                "id": int(item.get("id", 0)),
                "question": str(item.get("question", "")),
                "answer": ans_float,
                "relevant_docs": list(item.get("relevant_docs", [])),
                "relevant_tables": list(item.get("relevant_tables", [])),
                "evidence": list(item.get("evidence", [])),
                "pandas_query": str(item.get("pandas_query", "df1.iloc[0]"))
            }
            formatted_predictions.append(formatted_item)

        # Write submission.json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_predictions, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(formatted_predictions)} predictions to {json_path}")

        # Create submission.zip
        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_path, arcname="submission.json")

            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.output_dir)
                    zipf.write(full_path, arcname=rel_path)

        print(f"Created submission package successfully: {zip_filepath}")
        return zip_filepath
