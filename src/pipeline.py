"""
VERITAS Pipeline
Wires all agents into a single pipeline.run(claim) call.

Flow:
1. Decomposer — split claim into sub-claims, filter opinions
2. EvidenceHunter — retrieve passages per sub-claim
3. ContradictionDetector — check for conflicting evidence
4. Verifier — verdict per sub-claim (Ollama)
5. Synthesizer — aggregate into final verdict
"""
import sys
sys.path.insert(0, ".")
import time
import gc
from src.decomposer import Decomposer
from src.retriever import EvidenceHunter
from src.contradiction_detector import ContradictionDetector
from src.verifier import Verifier
from src.synthesizer import Synthesizer


class VERITASPipeline:

    def __init__(self):
        print("[Pipeline] Initializing VERITAS...")
        self.decomposer = Decomposer()
        self.hunter = EvidenceHunter()
        self.contradiction_detector = ContradictionDetector()
        self.verifier = Verifier()
        self.synthesizer = Synthesizer()
        print("[Pipeline] Ready.")

    def run(self, claim: str) -> dict:
        """
        Full pipeline: claim string → final verdict dict.
        """
        start_time = time.time()

        # Step 1: Decompose
        decomp = self.decomposer.process(claim)

        if not decomp["is_factual"]:
            return {
                "claim": claim,
                "verdict": "NOT_FACTUAL",
                "confidence": 0.0,
                "explanation": f"Claim is not fact-checkable: {decomp['reason']}",
                "subclaims": [],
                "evidence": {},
                "contradiction_detected": False,
                "latency": round(time.time() - start_time, 2)
            }

        subclaims = [sc["text"] for sc in decomp["subclaims"]]

        # Step 2: Retrieve evidence per sub-claim
        evidence_map = {}
        for subclaim in subclaims:
            evidence_map[subclaim] = self.hunter.retrieve(subclaim, top_k=10)

        # Step 3: Unload embedder to free RAM before Ollama
        self.hunter.unload_embedder()
        gc.collect()

        # Step 4: Contradiction detection
        all_passages = []
        for passages in evidence_map.values():
            all_passages.extend(passages)
        contradiction_result = self.contradiction_detector.detect(all_passages)

        # Step 5: Verify each sub-claim
        subclaim_results = []
        for subclaim in subclaims:
            evidence = evidence_map[subclaim]
            result = self.verifier.verify(subclaim, evidence)
            subclaim_results.append(result)

        # Step 6: Synthesize final verdict
        final = self.synthesizer.synthesize(
            subclaim_results,
            contradiction_detected=contradiction_result.has_contradiction
        )

        return {
            "claim": claim,
            "verdict": final.verdict,
            "confidence": final.confidence,
            "explanation": final.explanation,
            "subclaims": subclaim_results,
            "evidence": {
                sc: [p["text"] for p in evidence_map[sc][:3]]
                for sc in subclaims
            },
            "contradiction_detected": contradiction_result.has_contradiction,
            "conflict_score": contradiction_result.conflict_score,
            "latency": round(time.time() - start_time, 2)
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    pipeline = VERITASPipeline()

    test_claims = [
        "Barack Obama was born in Hawaii.",
        "The Eiffel Tower was built in 1889 and is located in Paris.",
        "I think climate change is a hoax.",
    ]

    print("\n" + "=" * 60)
    print("VERITAS Pipeline Test")
    print("=" * 60)

    for claim in test_claims:
        print(f"\nClaim: {claim}")
        result = pipeline.run(claim)
        print(f"  Verdict:       {result['verdict']}")
        print(f"  Confidence:    {result['confidence']}")
        print(f"  Explanation:   {result['explanation']}")
        print(f"  Contradiction: {result['contradiction_detected']}")
        print(f"  Latency:       {result['latency']}s")
        if result['subclaims']:
            print(f"  Sub-claims:    {len(result['subclaims'])}")
            for sc in result['subclaims']:
                print(f"    → {sc['verdict']} ({sc['confidence']}) | {sc['claim'][:60]}")