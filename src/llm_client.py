import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

try:
    from config.config import V2EQUERY_MODEL, T2PANDAS_MODEL
except ImportError:
    V2EQUERY_MODEL = "Qwen/Qwen3-8B"
    T2PANDAS_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


class LLMClient:
    """
    Local/Remote LLM Client supporting:
    - Silent offline local execution with fast heuristic fallback
    - Local Ollama / vLLM / LMStudio / OpenAI-compatible API
    - Direct HuggingFace pipeline execution if GPU is available
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_hf_direct: bool = False,
        enabled: Optional[bool] = None
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", None)
        self.use_hf_direct = use_hf_direct or (os.getenv("USE_HF_DIRECT", "0") == "1")
        
        # If enabled is not explicitly set, enable if base_url is explicitly given or env var is set
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = bool(self.base_url) or self.use_hf_direct or (os.getenv("ENABLE_LLM", "0") == "1")

        self.hf_pipelines = {}
        self._warned_offline = False

    def generate(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Generates completion using API endpoint, direct HF pipeline, or falls back to local heuristic.
        """
        if not self.enabled:
            return ""

        # 1. Direct HuggingFace Pipeline (for local GPU execution)
        if self.use_hf_direct:
            hf_res = self._call_hf_direct(model_name, system_prompt, user_prompt, temperature)
            if hf_res:
                return hf_res

        # 2. Try OpenAI-compatible API (FastAPI / vLLM / LMStudio / Ollama)
        if self.base_url:
            res = self._call_openai_compatible(model_name, system_prompt, user_prompt, temperature)
            if res:
                return res

            res_ollama = self._call_ollama(model_name, system_prompt, user_prompt, temperature)
            if res_ollama:
                return res_ollama

        if not self._warned_offline:
            print(f"  [LLM Client] LLM server not reachable at '{self.base_url}'. Seamlessly using Local Heuristic Engine.")
            self._warned_offline = True
            self.enabled = False

        return ""

    def _call_openai_compatible(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        url = self.base_url
        if not url.startswith("http"):
            url = f"http://{url}"

        endpoint = f"{url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    choices = resp_json.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return None

    def _call_ollama(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        url = self.base_url
        if not url.startswith("http"):
            url = f"http://{url}"

        endpoint = f"{url}/api/generate"
        headers = {"Content-Type": "application/json"}

        prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    return resp_json.get("response", "").strip()
        except Exception:
            pass
        return None

    def _call_hf_direct(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        if model_name not in self.hf_pipelines:
            try:
                import torch
                from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
                print(f"Loading local HuggingFace pipeline for {model_name}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
                self.hf_pipelines[model_name] = pipeline("text-generation", model=model, tokenizer=tokenizer)
            except Exception as e:
                print(f"Failed to load HF model {model_name}: {e}")
                return None

        pipe = self.hf_pipelines.get(model_name)
        if not pipe:
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            out = pipe(messages, max_new_tokens=512, do_sample=(temperature > 0))
            gen = out[0]["generated_text"]
            if isinstance(gen, list):
                return gen[-1].get("content", "").strip()
            return str(gen).strip()
        except Exception:
            return None
