# VERITAS — Project Overview for Claude

## What is VERITAS?

VERITAS is a multi-agent RAG-based fact-checking system. Given a natural language claim, it retrieves evidence from a Wikipedia-derived knowledge base and returns a structured verdict (SUPPORTED / REFUTED / PARTIALLY_SUPPORTED / CONFLICTING / INSUFFICIENT_EVIDENCE).

---

## Architecture

```
Claim
  │
  ▼
[Decomposer]          src/decomposer.py
  Rule-based claim splitter + opinion filter.
  Outputs atomic sub-claims, rejects opinion/non-factual claims.
  │
  ▼
[EvidenceHunter]      src/retriever.py
  ChromaDB + all-MiniLM-L6-v2 embeddings.
  Retrieves top-k passages per sub-claim from FEVER wiki corpus.
  Unloads embedder from RAM before calling Ollama (RAM management).
  │
  ▼
[ContradictionDetector]   src/contradiction_detector.py
  Checks retrieved passages for conflicting statements.
  Flags value conflicts and negation conflicts.
  │
  ▼
[Verifier]            src/verifier.py
  Calls Ollama (qwen3.5:0.8b) with a structured prompt.
  Returns verdict + confidence + explanation per sub-claim.
  Falls back to DistilBERT classifier if confidence < 0.65.
  │
  ▼
[Synthesizer]         src/synthesizer.py
  Rule-based aggregation of sub-claim verdicts.
  No model — pure logic (majority vote, conflict detection).
  │
  ▼
Final Verdict Dict
```

---

## Tech Stack

| Component        | Technology                              |
|------------------|-----------------------------------------|
| Vector store     | ChromaDB (persistent, cosine similarity)|
| Embeddings       | sentence-transformers/all-MiniLM-L6-v2  |
| LLM (local)      | Ollama — qwen3.5:0.8b                   |
| Kaggle baseline  | cross-encoder/nli-deberta-v3-small (NLI)|
| Evaluation data  | FEVER dev set (dev.csv)                 |
| Metrics          | scikit-learn classification_report      |
| API              | FastAPI (api/main.py)                   |
| UI               | Streamlit (ui/app.py)                   |

---

## File Structure

```
veritas/
├── src/
│   ├── decomposer.py          Agent 1 — claim splitter + factuality filter
│   ├── retriever.py           Agent 2 — ChromaDB evidence retrieval
│   ├── contradiction_detector.py  Agent 3 — conflicting evidence detection
│   ├── verifier.py            Agent 4 — Ollama verdict generation
│   ├── synthesizer.py         Agent 5 — sub-claim aggregation
│   ├── pipeline.py            Wires all agents; main entry point
│   └── evaluator.py           FEVER dev set evaluation harness
├── kaggle/
│   └── baseline_eval.py       Self-contained Kaggle T4 evaluator (no Ollama)
├── data/
│   ├── build_index.py         Builds ChromaDB index from FEVER wiki-pages
│   ├── download_datasets.py   Downloads FEVER dataset
│   └── fever/                 FEVER dev.csv + wiki-pages/
├── api/
│   └── main.py                FastAPI REST endpoint
├── ui/
│   └── app.py                 Streamlit demo UI
├── logs/
│   ├── evaluations.jsonl      Running log of pipeline calls
│   └── eval_results/          Per-run metrics + predictions JSON
├── evaluation/                Placeholder for future evaluation modules
├── requirements.txt
├── .env                       CHROMA_PATH, OLLAMA_BASE_URL, OLLAMA_MODEL
└── CLAUDE.md                  This file
```

---

## Key Design Decisions

- **RAM management**: `EvidenceHunter.unload_embedder()` is called before Ollama inference to avoid OOM on low-RAM machines (both MiniLM and qwen3.5 loaded simultaneously = ~2GB+).
- **Verdict mapping**: FEVER uses SUPPORTS/REFUTES/NOT ENOUGH INFO; VERITAS uses 5 internal verdicts that map back to 3-class for evaluation (see `VERITAS_TO_3CLASS` in evaluator.py).
- **Balanced sampling**: Evaluation always samples equally across 3 FEVER classes to avoid class-imbalance distorting Macro-F1.
- **Kaggle baseline**: Uses NLI cross-encoder instead of Ollama since Kaggle T4 notebooks have no Ollama. Scores are directly comparable (3-class classification on FEVER dev).

---

## Running Locally

```bash
# Build index (one-time)
python data/build_index.py

# Run pipeline on a single claim
python src/pipeline.py

# Run evaluation (30 claims)
python src/evaluator.py

# Kaggle baseline (requires FEVER data)
python kaggle/baseline_eval.py --fever_dir ./data/fever --n_samples 300
```

---

## Environment Variables (`.env`)

```
CHROMA_PATH=./data/chroma_index
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:0.8b
```

---

## Development Log

| Day | Milestone |
|-----|-----------|
| 1–2 | Project scaffold, ChromaDB index, MiniLM retrieval |
| 3   | All 5 agents built and wired (decomposer → synthesizer) |
| 4   | FEVER evaluation harness, precision/recall/F1 |
| 5   | Evaluator bug fixed; Kaggle NLI baseline evaluator |
