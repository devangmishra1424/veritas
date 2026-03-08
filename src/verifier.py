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
        # Fallback slot — filled after Kaggle training Day 16
        self.fallback = fallback_classifier
        print(f"[Verifier] Ready. Model: {self.model}")

    def _build_prompt(self, claim: str, evidence: list[dict]) -> str:
        """Build prompt with claim and retrieved evidence."""

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
        """Call Ollama and parse response."""
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
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
            raise ValueError(f"Unexpected format: {list(response_json.keys())}")

        # Strip thinking tags
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        # Clean markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        return result

    def verify(self, claim: str, evidence: list[dict]) -> dict:
        """
        Main verification method.
        Returns verdict dict with routing info.
        """
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

            # Route to fallback if confidence too low
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
    import sys
    sys.path.insert(0, ".")
    from src.retriever import EvidenceHunter

    print("=" * 60)
    print("Verifier Agent Test")
    print("=" * 60)

    hunter = EvidenceHunter()
    verifier = Verifier()

    test_claims = [
        "Barack Obama was born in Hawaii.",
        "The Great Wall of China is visible from space.",
        "Einstein won the Nobel Prize in Physics in 1921.",
    ]

    for claim in test_claims:
        print(f"\nClaim: {claim}")
        evidence = hunter.retrieve(claim, top_k=5)
        print(f"Evidence retrieved: {len(evidence)} passages")
        result = verifier.verify(claim, evidence)
        print(f"  Verdict:     {result['verdict']}")
        print(f"  Confidence:  {result['confidence']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"  Latency:     {result['latency']}s")
        print(f"  Fallback:    {result['routed_to_fallback']}")