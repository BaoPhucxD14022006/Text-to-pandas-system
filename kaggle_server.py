# Kaggle Server Code - Deploy LLM API Server on Kaggle GPU T4 x2

import os
import sys
import time
import threading
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Kaggle LLM API Server")
MODELS = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Kaggle LLM Server is running"}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    model_name = req.model
    pipe = MODELS.get(model_name)

    if not pipe and MODELS:
        pipe = list(MODELS.values())[0]

    if not pipe:
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    messages_input = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        output = pipe(messages_input, max_new_tokens=512, do_sample=(req.temperature > 0))
        gen_text = output[0]["generated_text"]
        if isinstance(gen_text, list):
            res_content = gen_text[-1].get("content", "").strip()
        else:
            res_content = str(gen_text).strip()
    except Exception as e:
        print(f"Generation error: {e}")
        res_content = ""

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": res_content
                }
            }
        ]
    }

def load_all_models():
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print("\n🔄 Đang tải các mô hình AI lên GPU Kaggle (Qwen & DeepSeek)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    models_to_load = {
        "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    }

    for name, path in models_to_load.items():
        try:
            print(f"⏳ Đang load: {name} ...")
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            MODELS[name] = pipeline("text-generation", model=model, tokenizer=tokenizer)
            print(f"✅ Đã load thành công: {name}")
        except Exception as e:
            print(f"❌ Lỗi load {name}: {e}")
