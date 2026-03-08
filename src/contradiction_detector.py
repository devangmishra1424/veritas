"""
VERITAS Contradiction Detector
Agent 3: Checks if retrieved evidence passages contradict each other.
Uses simple NLI-style heuristics now.
Zero-shot NLI model (cross-encoder/nli-deberta-v3-small) added after Day 14.
"""

from dataclasses import dataclass


@dataclass
class ContradictionResult:
    has_contradiction: bool
    conflicting_pairs: list[tuple[int, int]]
    conflict_score: float
    explanation: str


class ContradictionDetector:

    # Word pairs that signal contradiction between passages
    CONTRADICTION_PAIRS = [
        ("was born in", "was not born in"),
        ("is true", "is false"),
        ("did", "did not"),
        ("is", "is not"),
        ("won", "lost"),
        ("supported", "opposed"),
        ("confirmed", "denied"),
        ("exists", "does not exist"),
        ("visible", "not visible"),
        ("built in", "not built in"),
    ]

    # Negation words that flip meaning
    NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "cannot", "wasn't",
                      "isn't", "aren't", "didn't", "doesn't", "don't", "won't"}

    def __init__(self):
        print("[ContradictionDetector] Ready. Mode: heuristic (NLI model added Day 14)")

    def _has_negation_conflict(self, text_a: str, text_b: str) -> bool:
        """Check if one passage negates the other using keyword heuristics."""
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        a_words = set(a_lower.split())
        b_words = set(b_lower.split())

        a_negated = bool(a_words & self.NEGATION_WORDS)
        b_negated = bool(b_words & self.NEGATION_WORDS)

        # One negated, one not — potential contradiction
        if a_negated != b_negated:
            # Check for shared content words (ignoring stopwords)
            stopwords = {"the", "a", "an", "in", "on", "at", "to", "for",
                         "of", "and", "or", "but", "was", "is", "are", "were"}
            a_content = a_words - stopwords - self.NEGATION_WORDS
            b_content = b_words - stopwords - self.NEGATION_WORDS
            overlap = a_content & b_content
            if len(overlap) >= 2:
                return True

        return False

    def _check_pair(self, passage_a: dict, passage_b: dict) -> bool:
        """Check if two passages contradict each other."""
        text_a = passage_a["text"]
        text_b = passage_b["text"]

        # Check known contradiction patterns
        for pos_phrase, neg_phrase in self.CONTRADICTION_PAIRS:
            a_has_pos = pos_phrase in text_a.lower()
            b_has_neg = neg_phrase in text_b.lower()
            b_has_pos = pos_phrase in text_b.lower()
            a_has_neg = neg_phrase in text_a.lower()

            if (a_has_pos and b_has_neg) or (b_has_pos and a_has_neg):
                return True

        # Check same predicate, different object (e.g. "born in X" vs "born in Y")
        PREDICATES = ["born in", "died in", "located in", "founded in",
                      "invented by", "won in", "occurred in", "created in"]
        for predicate in PREDICATES:
            if predicate in text_a.lower() and predicate in text_b.lower():
                # Extract what follows the predicate
                a_val = text_a.lower().split(predicate)[-1].strip().split()[0]
                b_val = text_b.lower().split(predicate)[-1].strip().split()[0]
                if a_val and b_val and a_val != b_val:
                    return True

        # Check negation conflict
        if self._has_negation_conflict(text_a, text_b):
            return True

        return False

    def detect(self, passages: list[dict]) -> ContradictionResult:
        """
        Check all passage pairs for contradictions.
        Returns ContradictionResult with conflict details.
        """
        if len(passages) < 2:
            return ContradictionResult(
                has_contradiction=False,
                conflicting_pairs=[],
                conflict_score=0.0,
                explanation="Not enough passages to detect contradiction."
            )

        conflicting_pairs = []

        for i in range(len(passages)):
            for j in range(i + 1, len(passages)):
                if self._check_pair(passages[i], passages[j]):
                    conflicting_pairs.append((i, j))

        total_pairs = len(passages) * (len(passages) - 1) / 2
        conflict_score = len(conflicting_pairs) / total_pairs if total_pairs > 0 else 0.0

        has_contradiction = len(conflicting_pairs) > 0

        if has_contradiction:
            pair_desc = ", ".join([f"[{a+1}] vs [{b+1}]" for a, b in conflicting_pairs])
            explanation = f"Conflicting evidence found in passages: {pair_desc}"
        else:
            explanation = "No contradictions detected among retrieved passages."

        return ContradictionResult(
            has_contradiction=has_contradiction,
            conflicting_pairs=conflicting_pairs,
            conflict_score=round(conflict_score, 3),
            explanation=explanation
        )


if __name__ == "__main__":
    detector = ContradictionDetector()

    print("=" * 60)
    print("Contradiction Detector Test")
    print("=" * 60)

    test_cases = [
        {
            "label": "Clear contradiction",
            "passages": [
                {"text": "Barack Obama was born in Hawaii. [Source: Barack_Obama]"},
                {"text": "Barack Obama was born in Kenya. [Source: Barack_Obama]"},
                {"text": "Barack Obama resides in Washington D.C. [Source: Barack_Obama]"},
            ]
        },
        {
            "label": "No contradiction",
            "passages": [
                {"text": "The Eiffel Tower is located in Paris. [Source: Eiffel_Tower]"},
                {"text": "The Eiffel Tower was built in 1889. [Source: Eiffel_Tower]"},
                {"text": "The Eiffel Tower is 330 meters tall. [Source: Eiffel_Tower]"},
            ]
        },
        {
            "label": "Negation conflict",
            "passages": [
                {"text": "The Great Wall of China is visible from space."},
                {"text": "The Great Wall of China is not visible from space with the naked eye."},
            ]
        }
    ]

    for case in test_cases:
        print(f"\nTest: {case['label']}")
        result = detector.detect(case["passages"])
        print(f"  Has contradiction: {result.has_contradiction}")
        print(f"  Conflict score:    {result.conflict_score}")
        print(f"  Explanation:       {result.explanation}")