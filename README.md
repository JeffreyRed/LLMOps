# 🤖 Agentic RAG Pipeline

> An **LLMOps** project demonstrating an agentic Retrieval-Augmented Generation (RAG) pipeline with full observability — built with free tools and open-source models.

![CI](https://github.com/YOUR_USERNAME/agentic-rag-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA%203.3%2070B-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![LangSmith](https://img.shields.io/badge/Observability-LangSmith-yellow)

---

## 📌 What is this?

This project implements an **agentic RAG pipeline** — the agent decides:
- ✅ **Whether** to retrieve documents or answer from memory
- ✅ **What** query to send to the vector store
- ✅ **When** it has enough context to answer confidently

All agent decisions are **traced and logged** to LangSmith for full LLMOps observability.
Every push to `main` automatically runs linting, unit tests, and a prompt evaluation gate via GitHub Actions CI.

---

## 🧱 Architecture

```
User Question
      │
      ▼
 ┌─────────────┐
 │  LLM Agent  │  ← Groq (LLaMA 3.3 70B) via LangChain
 └──────┬──────┘
        │ decides to call tool?
        ▼
 ┌─────────────────────────┐
 │  search_knowledge_base  │  ← Tool: ChromaDB retriever
 └──────────┬──────────────┘
            │ top-5 relevant chunks (MMR)
            ▼
 ┌─────────────┐
 │  LLM Agent  │  ← synthesizes answer from retrieved context
 └─────────────┘
        │
        ▼
    Final Answer
        │
        ▼
 ┌─────────────┐
 │  LangSmith  │  ← traces every step for observability
 └─────────────┘
```

---

## 🛠️ Tech Stack (all free!)

| Component | Tool | Why |
|---|---|---|
| LLM | [Groq](https://console.groq.com) + LLaMA 3.3 70B | Free API, ultra-fast inference |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Local, no API cost |
| Vector Store | ChromaDB | Local, persistent, no server needed |
| Orchestration | LangChain | Industry-standard agent framework |
| Observability | LangSmith | Traces every agent decision |
| CI/CD | GitHub Actions | Auto-runs evals on every push |
| Dataset | HuggingFace `wikipedia` | Free, no login needed |

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-pipeline.git
cd agentic-rag-pipeline

# With conda (recommended)
conda create -n agentic-rag python=3.11 -y
conda activate agentic-rag
pip install -r requirements.txt

# Or with venv
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up your API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
# - GROQ_API_KEY      → https://console.groq.com       (free)
# - LANGCHAIN_API_KEY → https://smith.langchain.com    (free)
```

> ⚠️ `.env` is git-ignored and will never be committed. See [CI_SETUP.md](./CI_SETUP.md) for how to add secrets to GitHub Actions.

### 3. Ingest the Wikipedia dataset

```bash
python ingest.py
```

Downloads 200 Wikipedia articles, chunks them, embeds them locally, and saves to ChromaDB. Takes ~2-3 minutes on first run.

### 4. Run the agent

```bash
python agent.py
```

### 5. Run evaluations

```bash
python evaluate.py
```

Generates `data/eval_report.json` with keyword scores and latency metrics per question.

---

## 📁 Project Structure

```
agentic-rag-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions — runs on every push
├── tests/
│   └── test_pipeline.py    # Unit tests (chunking, scoring, config)
├── data/
│   ├── chroma_db/          # Persisted vector store (git-ignored)
│   └── eval_report.json    # Latest evaluation results
├── agent.py                # Agentic RAG — main entry point
├── config.py               # Loads .env settings safely
├── ingest.py               # Dataset loading, chunking, embedding
├── retriever.py            # ChromaDB retriever as LangChain tool
├── evaluate.py             # LLMOps evaluation suite
├── pytest.ini              # Test configuration
├── .env                    # Your secrets — never committed ⚠️
├── .env.example            # Template — safe to commit ✅
├── .gitignore              # Blocks .env and chroma_db/
├── CI_SETUP.md             # How to add GitHub secrets for CI
├── environment.yml         # Conda environment (reproducible)
└── requirements.txt        # Python dependencies
```

---

## 🔍 LLMOps Concepts Demonstrated

| Concept | Where |
|---|---|
| **Agentic decision-making** | Agent chooses when/what to retrieve |
| **Prompt engineering** | System prompt in `agent.py` |
| **Observability / Tracing** | LangSmith traces every run |
| **Evaluation pipeline** | `evaluate.py` — keyword scoring + latency |
| **Config & secrets management** | `.env` + `config.py` — no hardcoded keys |
| **Vector store lifecycle** | Ingest → Embed → Query → Retrieve |
| **CI/CD quality gate** | GitHub Actions blocks merges if eval score drops |
| **Reproducibility** | `environment.yml` + `requirements.txt` |

---

## ⚙️ CI/CD Pipeline

Every `git push` to `main` triggers 3 automated jobs:

```
Push → GitHub Actions
         │
         ├── 1. lint    → Black (auto-format) + Flake8 (errors)
         │                ↓ pass
         ├── 2. test    → pytest unit tests (no API calls, fast)
         │                ↓ pass
         └── 3. eval    → runs agent on test questions
                          fails if avg score < 0.35 ← quality gate
```

### ✅ Latest Test Results

| Job | Status | Details |
|---|---|---|
| 🔍 Lint | ![pass](https://img.shields.io/badge/black-formatted-black) ![pass](https://img.shields.io/badge/flake8-passing-green) | Auto-formatted + no errors |
| 🧪 Unit Tests | ![pass](https://img.shields.io/badge/pytest-14%20passed-brightgreen) | Chunking, scoring, config, quality gate |
| 📊 Eval | ![pass](https://img.shields.io/badge/avg%20score-%3E0.35-brightgreen) | 5 questions · keyword scoring · latency logged |

> Live CI status is shown by the badge at the top of this README.
> Every run saves an `eval_report.json` artifact downloadable from the [Actions tab](../../actions).

See [CI_SETUP.md](./CI_SETUP.md) for setup instructions.

---

## 📸 LangSmith Trace Example

After running the agent, go to [smith.langchain.com](https://smith.langchain.com) → project `agentic-rag-pipeline`:

```
Run: "What causes earthquakes?"
  ├── LLM decides → call tool: search_wikipedia_knowledge_base
  ├── Tool input: "earthquake tectonic plates cause"
  ├── Retrieved 5 chunks (0.3s)
  └── LLM synthesizes final answer (0.8s)
Total: 1.1s | Tokens: 842
```

---

## 🤝 Contributing

PRs welcome! Ideas for extensions:
- Add a web UI with Streamlit
- Swap in a different dataset (financial news, medical FAQ)
- Add RAGAS-based evaluation metrics (faithfulness, relevancy)
- Implement prompt version tracking with MLflow

---

## 📄 License

MIT