"""
VERITAS Verifier Agent
Agent 4: Takes a sub-claim + evidence passages and produces a verdict.
Primary: Ollama local model (qwen3.5:0.8b)
Fallback: DistilBERT classifier (added after Kaggle training in Day 16)
"""

import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")

VALID_VERDICTS = {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
CONFIDENCE_THRESHOLD = 0.65


class Verifier:

    def __init__(self, fallback_classifier=None):
        self.ollama_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.fallback = fallback_classifier
        print(f"[Verifier] Ready. Model: {self.model}")

    def _build_prompt(self, claim: str, evidence: list[dict]) -> str:
        if evidence:
            evidence_text = "\n".join([
                f"[{i+1}] {e['text'][:300]}"
                for i, e in enumerate(evidence[:5])
            ])
        else:
            evidence_text = "No evidence retrieved."

        return f"""You are a fact-checking assistant. Given a claim and evidence passages, determine if the evidence supports or refutes the claim.

CLAIM: {claim}

EVIDENCE:
{evidence_text}

Respond with ONLY a valid JSON object with these exact fields:
- verdict: one of SUPPORTS, REFUTES, NOT_ENOUGH_INFO
- confidence: float between 0.0 and 1.0
- explanation: one sentence explaining your reasoning

Rules:
- Use SUPPORTS if evidence confirms the claim
- Use REFUTES if evidence contradicts the claim
- Use NOT_ENOUGH_INFO if evidence is insufficient or unrelated
- Be conservative — only use SUPPORTS or REFUTES when evidence is clear

JSON only. No markdown, no code fences, no extra text."""

    def _call_ollama(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "presence_penalty": 1.5,
                    "num_predict": 200
                }
            },
            timeout=120
        )

        response_json = response.json()

        if "message" in response_json:
            raw = response_json["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected format: {list(response_json.keys())}")

        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        return result

    def verify(self, claim: str, evidence: list[dict]) -> dict:
        start_time = time.time()
        prompt = self._build_prompt(claim, evidence)

        try:
            result = self._call_ollama(prompt)

            verdict = result.get("verdict", "NOT_ENOUGH_INFO").upper()
            if verdict not in VALID_VERDICTS:
                verdict = "NOT_ENOUGH_INFO"

            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            explanation = result.get("explanation", "No explanation.")

            routed_to_fallback = False
            if confidence < CONFIDENCE_THRESHOLD and self.fallback:
                fallback_result = self.fallback.classify(claim, evidence)
                verdict = fallback_result["verdict"]
                confidence = fallback_result["confidence"]
                explanation = fallback_result["explanation"]
                routed_to_fallback = True

            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation,
                "latency": round(time.time() - start_time, 2),
                "model": self.model,
                "routed_to_fallback": routed_to_fallback,
                "evidence_count": len(evidence)
            }

        except Exception as e:
            return {
                "claim": claim,
                "verdict": "NOT_ENOUGH_INFO",
                "confidence": 0.0,
                "explanation": f"Verifier error: {str(e)}",
                "latency": round(time.time() - start_time, 2),
                "model": self.model,
                "routed_to_fallback": False,
                "evidence_count": len(evidence)
            }


if __name__ == "__main__":
    verifier = Verifier()
    result = verifier.verify(
        "Barack Obama was born in Hawaii.",
        [{"text": "Barack Obama was born in Kenya. [Source: Barack_Obama]", "score": 0.78}]
    )
    print(result)