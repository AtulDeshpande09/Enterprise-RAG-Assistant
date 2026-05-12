from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Prompt(BaseModel):
    text: str

@app.post("/chat")
def chat(prompt: Prompt):

    inputs = tokenizer(prompt.text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}

import nest_asyncio
import uvicorn
from threading import Thread

nest_asyncio.apply()

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

Thread(target=run).start()

from pyngrok import ngrok

public_url = ngrok.connect(8000)

print(public_url)