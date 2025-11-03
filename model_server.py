"""FastAPI model server exposing /predict, /health and Prometheus metrics.
Uses a Hugging Face transformers model loaded into memory.
"""
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

MODEL_NAME = os.getenv('MODEL_NAME', 'gpt2')
app = FastAPI(title='LLMOps Model Server')

REQUEST_COUNT = Counter('llm_requests_total', 'Total inference requests', ['endpoint'])
REQUEST_LATENCY = Histogram('llm_request_latency_seconds', 'Inference latency seconds', ['endpoint'])

class InferenceRequest(BaseModel):
    input: str
    max_new_tokens: int = 50

@app.on_event('startup')
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print(f'Loaded model {MODEL_NAME}')

@app.get('/')
def root():
    return {'message': 'Welcome to the LLMOps Model Server! See /docs for API documentation.'}

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(req: InferenceRequest):
    start = time.time()
    REQUEST_COUNT.labels(endpoint='/predict').inc()
    try:
        inputs = tokenizer(req.input, return_tensors='pt')
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return {'output': text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start)

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)