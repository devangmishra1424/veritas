from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VERITAS API", version="0.1.0")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
LOG_PATH = os.getenv("LOG_PATH", "./logs")


class VerifyRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    claim: str
    use_fallback: Optional[bool] = False


class VerifyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    claim: str
    verdict: str
    confidence: float
    explanation: str
    latency_seconds: float
    model_used: str


@app.get("/health")
def health_check():
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags", timeout=3
        )
        ollama_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        ollama_status = "disconnected"

    return {
        "status": "healthy" if ollama_status == "connected" else "degraded",
        "ollama": ollama_status,
        "model": OLLAMA_MODEL,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/verify", response_model=VerifyResponse)
def verify_claim(request: VerifyRequest):
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    start_time = time.time()

    prompt = f"""You are a fact-checking assistant. Given the following claim,
respond with ONLY a JSON object with these exact fields:
- verdict: one of SUPPORTS, REFUTES, NOT_ENOUGH_INFO
- confidence: float between 0 and 1
- explanation: one sentence explaining why

Claim: {request.claim}

Respond with valid JSON only. No markdown, no code fences, no extra text."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "presence_penalty": 1.0,
                    "num_predict": 200
                }
            },
            timeout=120
        )

        response_json = response.json()

        if "response" in response_json:
            raw = response_json["response"].strip()
        elif "message" in response_json:
            raw = response_json["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected response format: {list(response_json.keys())}")

        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        verdict = result.get("verdict", "NOT_ENOUGH_INFO")
        confidence = float(result.get("confidence", 0.5))
        explanation = result.get("explanation", "No explanation provided.")

    except Exception as e:
        verdict = "NOT_ENOUGH_INFO"
        confidence = 0.0
        explanation = f"Pipeline error: {str(e)}"

    latency = round(time.time() - start_time, 2)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "claim": request.claim,
        "verdict": verdict,
        "confidence": confidence,
        "latency_seconds": latency,
        "model_used": OLLAMA_MODEL
    }

    os.makedirs(LOG_PATH, exist_ok=True)
    with open(f"{LOG_PATH}/evaluations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return VerifyResponse(
        claim=request.claim,
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        latency_seconds=latency,
        model_used=OLLAMA_MODEL
    )
