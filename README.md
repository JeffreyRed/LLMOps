# 🤖 Agentic RAG Pipeline

> An **LLMOps** project demonstrating an agentic Retrieval-Augmented Generation (RAG) pipeline with full observability — built with free tools and open-source models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
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
| Dataset | HuggingFace `wikipedia` | Free, no login needed |

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-pipeline.git
cd agentic-rag-pipeline
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up your API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# - GROQ_API_KEY from https://console.groq.com (free)
# - LANGCHAIN_API_KEY from https://smith.langchain.com (free)
```

### 3. Ingest the Wikipedia dataset

```bash
python src/ingest.py
```

This downloads 200 Wikipedia articles, chunks them, embeds them locally, and saves to ChromaDB. Takes ~2-3 minutes on first run.

### 4. Run the agent

```bash
python src/agent.py
```

### 5. Run evaluations

```bash
python src/evaluate.py
```

Generates a `data/eval_report.json` with scores and latency metrics.

---

## 📁 Project Structure

```
agentic-rag-pipeline/
├── src/
│   ├── config.py       # Loads .env settings
│   ├── ingest.py       # Dataset loading, chunking, embedding
│   ├── retriever.py    # ChromaDB retriever as LangChain tool
│   ├── agent.py        # Agentic RAG — main entry point
│   └── evaluate.py     # LLMOps evaluation suite
├── data/
│   ├── chroma_db/      # Persisted vector store (git-ignored)
│   └── eval_report.json
├── .env.example        # Template — copy to .env and fill keys
├── .gitignore          # .env is never committed
└── requirements.txt
```

---

## 🔍 LLMOps Concepts Demonstrated

| Concept | Where |
|---|---|
| **Agentic decision-making** | Agent chooses when/what to retrieve |
| **Prompt engineering** | System prompt in `agent.py` |
| **Observability / Tracing** | LangSmith traces every run |
| **Evaluation pipeline** | `evaluate.py` — keyword scoring + latency |
| **Config management** | `.env` + `config.py` — no hardcoded secrets |
| **Vector store lifecycle** | Ingest → Embed → Query → Retrieve |

---

## 📸 LangSmith Trace Example

After running the agent, go to [smith.langchain.com](https://smith.langchain.com) → your project `agentic-rag-pipeline` to see full traces like:

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
