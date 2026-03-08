"""
VERITAS Synthesizer
Final step: Aggregates sub-claim verdicts into a single final verdict.
Rule-based aggregation — no model needed.
"""

from dataclasses import dataclass


FINAL_VERDICTS = {
    "SUPPORTED",
    "REFUTED",
    "PARTIALLY_SUPPORTED",
    "CONFLICTING",
    "INSUFFICIENT_EVIDENCE"
}


@dataclass
class FinalVerdict:
    verdict: str
    confidence: float
    explanation: str
    subclaim_count: int
    supported_count: int
    refuted_count: int
    insufficient_count: int
    has_conflict: bool


class Synthesizer:

    def __init__(self):
        print("[Synthesizer] Ready.")

    def synthesize(
        self,
        subclaim_results: list[dict],
        contradiction_detected: bool = False
    ) -> FinalVerdict:
        """
        Aggregate sub-claim verdicts into final verdict.

        Rules:
        - All SUPPORTS → SUPPORTED
        - All REFUTES → REFUTED
        - Any contradiction detected → CONFLICTING
        - Mix of SUPPORTS + REFUTES → PARTIALLY_SUPPORTED
        - All NOT_ENOUGH_INFO → INSUFFICIENT_EVIDENCE
        - Majority wins for mixed SUPPORTS/NOT_ENOUGH_INFO
        """

        if not subclaim_results:
            return FinalVerdict(
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                explanation="No sub-claims to evaluate.",
                subclaim_count=0,
                supported_count=0,
                refuted_count=0,
                insufficient_count=0,
                has_conflict=False
            )

        supported = [r for r in subclaim_results if r["verdict"] == "SUPPORTS"]
        refuted = [r for r in subclaim_results if r["verdict"] == "REFUTES"]
        insufficient = [r for r in subclaim_results if r["verdict"] == "NOT_ENOUGH_INFO"]

        n = len(subclaim_results)
        n_supported = len(supported)
        n_refuted = len(refuted)
        n_insufficient = len(insufficient)

        # Average confidence of the winning verdict group
        def avg_confidence(group):
            if not group:
                return 0.0
            return round(sum(r["confidence"] for r in group) / len(group), 3)

        # Contradiction overrides everything
        if contradiction_detected and n_refuted > 0:
            verdict = "CONFLICTING"
            confidence = avg_confidence(refuted)
            explanation = "Evidence sources contradict each other on this claim."

        # All refuted
        elif n_refuted == n:
            verdict = "REFUTED"
            confidence = avg_confidence(refuted)
            explanation = f"All {n} sub-claims are refuted by evidence."

        # All supported
        elif n_supported == n:
            verdict = "SUPPORTED"
            confidence = avg_confidence(supported)
            explanation = f"All {n} sub-claims are supported by evidence."

        # All insufficient
        elif n_insufficient == n:
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.1
            explanation = "No relevant evidence found for any sub-claim."

        # Mix of supported and refuted
        elif n_supported > 0 and n_refuted > 0:
            verdict = "PARTIALLY_SUPPORTED"
            confidence = round((avg_confidence(supported) + avg_confidence(refuted)) / 2, 3)
            explanation = (
                f"{n_supported} sub-claim(s) supported, "
                f"{n_refuted} refuted, "
                f"{n_insufficient} insufficient."
            )

        # Mostly supported with some insufficient
        elif n_supported > n_insufficient:
            verdict = "SUPPORTED"
            confidence = round(avg_confidence(supported) * (n_supported / n), 3)
            explanation = (
                f"{n_supported} of {n} sub-claims supported, "
                f"{n_insufficient} had insufficient evidence."
            )

        # Mostly insufficient
        else:
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.2
            explanation = (
                f"Insufficient evidence for most sub-claims "
                f"({n_insufficient} of {n})."
            )

        return FinalVerdict(
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            subclaim_count=n,
            supported_count=n_supported,
            refuted_count=n_refuted,
            insufficient_count=n_insufficient,
            has_conflict=contradiction_detected
        )


if __name__ == "__main__":
    synthesizer = Synthesizer()

    print("=" * 60)
    print("Synthesizer Test")
    print("=" * 60)

    test_cases = [
        {
            "label": "All supported",
            "results": [
                {"verdict": "SUPPORTS", "confidence": 0.9},
                {"verdict": "SUPPORTS", "confidence": 0.85},
            ],
            "contradiction": False
        },
        {
            "label": "All refuted",
            "results": [
                {"verdict": "REFUTES", "confidence": 0.95},
            ],
            "contradiction": False
        },
        {
            "label": "Mixed — partially supported",
            "results": [
                {"verdict": "SUPPORTS", "confidence": 0.88},
                {"verdict": "REFUTES", "confidence": 0.91},
                {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.4},
            ],
            "contradiction": False
        },
        {
            "label": "Conflicting evidence",
            "results": [
                {"verdict": "SUPPORTS", "confidence": 0.75},
                {"verdict": "REFUTES", "confidence": 0.82},
            ],
            "contradiction": True
        },
        {
            "label": "Insufficient evidence",
            "results": [
                {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.3},
                {"verdict": "NOT_ENOUGH_INFO", "confidence": 0.25},
            ],
            "contradiction": False
        },
    ]

    for case in test_cases:
        print(f"\nTest: {case['label']}")
        result = synthesizer.synthesize(case["results"], case["contradiction"])
        print(f"  Verdict:     {result.verdict}")
        print(f"  Confidence:  {result.confidence}")
        print(f"  Explanation: {result.explanation}")