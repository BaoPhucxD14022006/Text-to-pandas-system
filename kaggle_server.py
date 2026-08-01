# Kaggle Server Code - Deploy LLM API Server on Kaggle GPU T4 x2
# Copy this script into a Kaggle Notebook and run!

"""
Kaggle Notebook Server Setup Instructions:
1. Open Kaggle Notebook -> Set GPU T4 x2 -> Internet ON.
2. Install packages in Kaggle cell:
   !pip install -q fastapi uvicorn pyngrok transformers accelerate bitsandbytes
3. Run this script cell. It will print your Public Ngrok URL:
   Public URL: https://xxxx.ngrok-free.app
4. Copy that URL to your local PC and run:
   python test_single_question.py --id 1 --server-url https://xxxx.ngrok-free.app
"""

import os
import sys
import threading
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

app = FastAPI(title="Kaggle LLM API Server for Text-to-Pandas")

# Global dict holding loaded HuggingFace pipelines
MODELS = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0

@app.on_event("startup")
def load_models():
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    
    print("=== LOADING MODELS ON KAGGLE GPU ===")
    
    models_to_load = {
        "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    }

    for name, path in models_to_load.items():
        try:
            print(f"Loading {name}...")
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                load_in_4bit=True, # Quantize 4-bit to fit both models in GPU VRAM
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            MODELS[name] = pipeline("text-generation", model=model, tokenizer=tokenizer)
            print(f"Successfully loaded {name}!")
        except Exception as e:
            print(f"Error loading {name}: {e}")

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    model_name = req.model
    pipe = MODELS.get(model_name)
    
    # Fallback to first available model if exact string not matched
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

def start_tunnel(port=8000, authtoken=None):
    from pyngrok import ngrok
    if authtoken:
        ngrok.set_auth_token(authtoken)
    public_url = ngrok.connect(port)
    print("\n" + "=" * 60)
    print(f"🚀 KAGGLE LLM SERVER IS LIVE!")
    print(f"🌐 PUBLIC SERVER URL: {public_url}")
    print("=" * 60 + "\n")
    return public_url

if __name__ == "__main__":
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
