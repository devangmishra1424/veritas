"""
VERITAS Decomposer Agent
Agent 1: Splits compound claims into atomic sub-claims.
Uses rule-based splitting + a simple factuality filter.
Fine-tuned boundary classifier will replace this in Day 17.
"""

import re
from dataclasses import dataclass


@dataclass
class SubClaim:
    text: str
    original_claim: str
    split_reason: str
    index: int


class Decomposer:

    # Conjunctions that indicate compound claims
    SPLIT_PATTERNS = [
        r'\s+and\s+',
        r'\s+but\s+',
        r'\s+however\s+',
        r'\s+although\s+',
        r'\s+while\s+',
        r';\s*',
    ]

    # Opinion markers — these claims can't be fact-checked
    OPINION_MARKERS = [
        'i think', 'i believe', 'i feel', 'in my opinion',
        'should', 'must', 'best', 'worst', 'greatest', 'most beautiful',
        'better than', 'worse than', 'unfair', 'wrong to'
    ]

    # Verifiable entity patterns
    YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
    NUMBER_PATTERN = re.compile(r'\b\d+\b')
    PROPER_NOUN_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

    def __init__(self):
        self.split_regex = re.compile(
            '|'.join(self.SPLIT_PATTERNS),
            re.IGNORECASE
        )

    def decompose(self, claim: str) -> list[SubClaim]:
        """Split compound claim into atomic sub-claims."""
        claim = claim.strip()
        if not claim:
            return []

        parts = [p.strip() for p in self.split_regex.split(claim)]
        parts = [p for p in parts if len(p) > 10]

        if len(parts) <= 1:
            return [SubClaim(
                text=claim,
                original_claim=claim,
                split_reason="atomic",
                index=0
            )]

        return [
            SubClaim(
                text=p,
                original_claim=claim,
                split_reason=f"conjunction_split",
                index=i
            )
            for i, p in enumerate(parts)
        ]

    def is_factual(self, claim: str) -> tuple[bool, str]:
        """
        Check if a claim is factual (verifiable) or opinion-based.
        Returns (is_factual, reason).
        """
        claim_lower = claim.lower()

        # Check for opinion markers
        for marker in self.OPINION_MARKERS:
            if marker in claim_lower:
                return False, f"opinion_marker: '{marker}'"

        # Check for verifiable entities
        has_year = bool(self.YEAR_PATTERN.search(claim))
        has_number = bool(self.NUMBER_PATTERN.search(claim))
        has_proper_noun = bool(self.PROPER_NOUN_PATTERN.search(claim))

        if not (has_year or has_number or has_proper_noun):
            return False, "no_verifiable_entities"

        return True, "factual"

    def process(self, claim: str) -> dict:
        """
        Full decomposition pipeline.
        Returns structured result with sub-claims and factuality check.
        """
        is_factual, reason = self.is_factual(claim)

        if not is_factual:
            return {
                "original_claim": claim,
                "is_factual": False,
                "reason": reason,
                "subclaims": [],
                "count": 0
            }

        subclaims = self.decompose(claim)

        return {
            "original_claim": claim,
            "is_factual": True,
            "reason": "factual",
            "subclaims": [
                {
                    "text": sc.text,
                    "split_reason": sc.split_reason,
                    "index": sc.index
                }
                for sc in subclaims
            ],
            "count": len(subclaims)
        }


if __name__ == "__main__":
    decomposer = Decomposer()

    test_claims = [
        "The Eiffel Tower was built in 1889 and is located in Paris.",
        "Barack Obama was born in Hawaii but grew up in Indonesia.",
        "I think climate change is the biggest problem.",
        "The Great Wall of China is visible from space.",
        "Einstein won the Nobel Prize in 1921 and worked at Princeton University.",
        "The vaccine causes autism.",
    ]

    print("=" * 60)
    print("Decomposer Agent Test")
    print("=" * 60)

    for claim in test_claims:
        result = decomposer.process(claim)
        print(f"\nClaim: {claim}")
        print(f"  Factual: {result['is_factual']} ({result['reason']})")
        if result['is_factual']:
            for sc in result['subclaims']:
                print(f"  [{sc['index']+1}] {sc['text']} ({sc['split_reason']})")