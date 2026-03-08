"""
VERITAS ChromaDB Index Builder
Indexes FEVER evidence sentences for retrieval.
"""

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

CHROMA_PATH = "./data/chroma_index"
BATCH_SIZE = 500


def build_fever_index():
    print("Loading FEVER train data...")
    df = pd.read_csv("./data/fever/train.csv")

    print(f"Total rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    df = df.dropna(subset=["evidence_wiki_url"]).copy()
    df = df[df["label"] != "NOT ENOUGH INFO"].copy()
    print(f"Rows with evidence: {len(df):,}")

    passages = []
    ids = []
    metadatas = []
    seen = set()

    for idx, row in df.iterrows():
        claim = str(row["claim"]).strip()
        wiki_url = str(row["evidence_wiki_url"]).strip()

        text = f"{claim} [Source: {wiki_url}]"

        if text not in seen and len(text) > 20:
            seen.add(text)
            uid = f"fever_{idx}"
            passages.append(text)
            ids.append(uid)
            metadatas.append({
                "source": "fever",
                "wiki_url": wiki_url[:200],
                "label": str(row["label"]),
                "claim": claim[:200]
            })

        if len(passages) >= 50000:
            break

    print(f"Extracted {len(passages):,} unique passages")
    return passages, ids, metadatas


def index_passages(passages, ids, metadatas):
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("veritas_knowledge")
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name="veritas_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Indexing {len(passages):,} passages in batches of {BATCH_SIZE}...")

    for i in range(0, len(passages), BATCH_SIZE):
        batch_passages = passages[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]

        embeddings = embedder.encode(
            batch_passages,
            show_progress_bar=False
        ).tolist()

        collection.add(
            documents=batch_passages,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta
        )

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Indexed {min(i + BATCH_SIZE, len(passages)):,} / {len(passages):,}")

    print(f"\nIndex built. Total passages: {collection.count():,}")
    return collection


def test_retrieval(collection):
    print("\nTesting retrieval...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    query = "The Eiffel Tower was built in 1889"
    embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    print(f"Query: '{query}'")
    for i, doc in enumerate(results["documents"][0]):
        print(f"  [{i+1}] {doc[:120]}")
    print("Retrieval OK")


if __name__ == "__main__":
    print("=" * 50)
    print("VERITAS ChromaDB Index Builder")
    print("=" * 50)

    passages, ids, metadatas = build_fever_index()
    collection = index_passages(passages, ids, metadatas)
    test_retrieval(collection)

    print("\nDone. Index saved to:", CHROMA_PATH)