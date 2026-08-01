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
    LLM Client supporting local/remote Ollama / vLLM / OpenAI-compatible endpoints (including ngrok),
    direct HuggingFace pipeline execution on GPU (Kaggle/Colab),
    and rule-based fallbacks for offline execution.
    """

    def __init__(
        self,
        base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        api_key: Optional[str] = os.getenv("LLM_API_KEY", None),
        use_hf_direct: bool = os.getenv("USE_HF_DIRECT", "0") == "1"
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.use_hf_direct = use_hf_direct
        self.hf_pipelines = {}

    def generate(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Generates completion using API endpoint, direct HF pipeline, or fallback.
        """
        # 1. Direct HuggingFace Pipeline (for Kaggle / Colab local GPU process)
        if self.use_hf_direct:
            hf_res = self._call_hf_direct(model_name, system_prompt, user_prompt, temperature)
            if hf_res:
                return hf_res

        # 2. Try OpenAI-compatible API (FastAPI / vLLM / LMStudio / ngrok)
        res = self._call_openai_compatible(model_name, system_prompt, user_prompt, temperature)
        if res:
            return res

        # 3. Try Ollama API (/api/generate)
        res = self._call_ollama(model_name, system_prompt, user_prompt, temperature)
        if res:
            return res

        # 4. Fallback: offline empty response signaling dry-run heuristics
        return ""

    def _call_openai_compatible(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        # Formulate full chat completions URL
        url = self.base_url
        if not url.startswith("http"):
            url = f"http://{url}"

        if not url.endswith("/chat/completions"):
            if url.endswith("/v1"):
                url = f"{url}/chat/completions"
            else:
                url = f"{url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",  # Essential for ngrok free tunnels
            "User-Agent": "TextToPandasClient/1.0"
        }
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
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    choices = result.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        if content:
                            print(f"  [LLM SERVER RESPONSE ({model_name})]: Received successfully!")
                            return content
        except Exception as e:
            # Silence connection errors to allow clean fallback if server is offline
            pass
        return None

    def _call_ollama(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        url = f"{self.base_url}/api/generate"
        if not url.startswith("http"):
            url = f"http://{url}"

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
        payload = {
            "model": model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": temperature}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    res = result.get("response", "").strip()
                    if res:
                        print(f"  [OLLAMA SERVER RESPONSE ({model_name})]: Received successfully!")
                        return res
        except Exception:
            pass
        return None

    def _call_hf_direct(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> Optional[str]:
        try:
            import torch
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return None

        if model_name not in self.hf_pipelines:
            try:
                print(f"Loading HuggingFace model directly on GPU: {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                self.hf_pipelines[model_name] = pipeline("text-generation", model=model, tokenizer=tokenizer)
            except Exception as e:
                print(f"Error loading HF model {model_name}: {e}")
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
            generated = out[0]["generated_text"]
            if isinstance(generated, list):
                return generated[-1].get("content", "").strip()
            elif isinstance(generated, str):
                return generated.strip()
        except Exception:
            pass
        return None
