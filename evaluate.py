"""
evaluate.py — LLMOps evaluation: test the agent on a set of questions
and log results (pass/fail + latency) to a JSON report.

Run: python src/evaluate.py
"""
import json
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table

from agent import build_agent

console = Console()

# ── Evaluation questions with expected keywords ───
EVAL_QUESTIONS = [
    {
        "question": "What is photosynthesis?",
        "expected_keywords": ["light", "chlorophyll", "glucose", "plant", "carbon dioxide"],
    },
    {
        "question": "Who was Albert Einstein and what is he known for?",
        "expected_keywords": ["relativity", "physics", "Nobel", "theory"],
    },
    {
        "question": "What causes earthquakes?",
        "expected_keywords": ["tectonic", "plates", "fault", "seismic"],
    },
    {
        "question": "What is the theory of evolution?",
        "expected_keywords": ["Darwin", "natural selection", "species", "adaptation"],
    },
    {
        "question": "How does the human immune system work?",
        "expected_keywords": ["antibody", "white blood cells", "pathogen", "immune"],
    },
]


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Simple keyword-overlap score (0.0 to 1.0)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 2)


def run_evaluation():
    console.print("[bold cyan]🧪 Running LLMOps Evaluation Suite...[/bold cyan]\n")
    agent = build_agent()

    results = []
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("Score", justify="center")
    table.add_column("Latency (s)", justify="center")
    table.add_column("Status", justify="center")

    for item in EVAL_QUESTIONS:
        question = item["question"]
        keywords = item["expected_keywords"]

        start = time.time()
        try:
            result = agent.invoke({"input": question, "chat_history": []})
            answer = result["output"]
            latency = round(time.time() - start, 2)
            score = keyword_score(answer, keywords)
            status = "✅ PASS" if score >= 0.4 else "⚠️ REVIEW"
        except Exception as e:
            answer = f"ERROR: {e}"
            latency = round(time.time() - start, 2)
            score = 0.0
            status = "❌ FAIL"

        results.append({
            "question": question,
            "answer": answer,
            "score": score,
            "latency_s": latency,
            "status": status,
        })

        table.add_row(question[:60], str(score), str(latency), status)

    console.print(table)

    # ── Save report ────────────────────────────────
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "llama-3.3-70b-versatile",
        "num_questions": len(EVAL_QUESTIONS),
        "avg_score": round(sum(r["score"] for r in results) / len(results), 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 2),
        "results": results,
    }

    report_path = Path("data/eval_report.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[green]📄 Report saved to {report_path}[/green]")
    console.print(f"[bold]Average score: {report['avg_score']} | Avg latency: {report['avg_latency_s']}s[/bold]")


if __name__ == "__main__":
    run_evaluation()
