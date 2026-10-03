import os

# Models (used when local/remote LLM inference is enabled)
T2PANDAS_MODEL = os.getenv("T2PANDAS_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
V2EQUERY_MODEL = os.getenv("V2EQUERY_MODEL", "Qwen/Qwen3-8B")

# Local Storage Paths
DEFAULT_QUESTIONS_PATH = "ViFinQA/questions/questions.jsonl"
DEFAULT_STOCK_PATH = "ViFinQA/code_stock.csv"
DEFAULT_CATALOG_PATH = "processed_data/table_catalog.json"
DEFAULT_CSV_DIR = "processed_data/csv"
DEFAULT_OUTPUT_DIR = "submission"
DEFAULT_ZIP_PATH = "submission.zip"