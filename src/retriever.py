"""
VERITAS Evidence Hunter
Agent 2: Retrieves relevant evidence passages for a given claim.
Uses ChromaDB + MiniLM for first-stage retrieval.
Reranker slot is prepared but uses score passthrough until
fine-tuned weights are available (added in Day 11).
"""

import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_index")


class EvidenceHunter:

    def __init__(self, reranker=None):
        print("[EvidenceHunter] Loading ChromaDB...")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection("veritas_knowledge")

        print("[EvidenceHunter] Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Reranker slot — None until fine-tuned weights available
        self.reranker = reranker

        count = self.collection.count()
        print(f"[EvidenceHunter] Ready. Index contains {count:,} passages.")

    def reload_embedder(self):
        """Reload embedding model if it was unloaded."""
        if self.embedder is None:
            print("[EvidenceHunter] Reloading embedder...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        First-stage retrieval: embed query, search ChromaDB.
        Returns top_k passages as list of dicts.
        """
        if not query.strip():
            return []

        # Auto-reload embedder if unloaded
        self.reload_embedder()

        embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        passages = []
        for i in range(len(results["documents"][0])):
            passages.append({
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "rank": i + 1
            })

        if self.reranker:
            passages = self.reranker.rerank(query, passages)

        return passages[:5]

    def retrieve_for_subclaims(self, subclaims: list[str]) -> dict:
        """
        Retrieve evidence for a list of sub-claims.
        Returns dict mapping each sub-claim to its top passages.
        """
        results = {}
        for subclaim in subclaims:
            results[subclaim] = self.retrieve(subclaim, top_k=10)
        return results

    def unload_embedder(self):
        """Free embedding model from RAM before calling Ollama."""
        del self.embedder
        self.embedder = None
        print("[EvidenceHunter] Embedder unloaded from RAM.")


if __name__ == "__main__":
    hunter = EvidenceHunter()

    test_claims = [
        "The Eiffel Tower was built in 1889.",
        "Barack Obama was born in Hawaii.",
        "The Great Wall of China is visible from space."
    ]

    print("\n" + "=" * 50)
    print("Evidence Hunter Test")
    print("=" * 50)

    for claim in test_claims:
        print(f"\nClaim: {claim}")
        passages = hunter.retrieve(claim, top_k=5)
        for p in passages:
            print(f"  [{p['rank']}] score={p['score']:.3f} | {p['text'][:100]}")