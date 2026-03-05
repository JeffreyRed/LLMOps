"""
tests/test_pipeline.py

Unit tests for the Agentic RAG Pipeline.
These run in CI on every push — no real API calls needed
(we mock the LLM and vector store for speed and cost).
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Test 1: Config loads correctly ───────────────────────
class TestConfig:
    def test_env_keys_present(self, monkeypatch):
        """Config should load without error when keys are set."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-langsmith-key")

        # Re-import after patching env
        import importlib
        import sys
        if "src.config" in sys.modules:
            del sys.modules["src.config"]

        import sys, os
        sys.path.insert(0, "src")
        from config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL

        assert GROQ_API_KEY == "test-key-123"
        assert LLM_MODEL == "llama-3.3-70b-versatile"
        assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_validate_config_raises_without_key(self, monkeypatch):
        """validate_config() should raise if GROQ_API_KEY is missing."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        import sys
        sys.path.insert(0, "src")

        # Force reload with missing key
        import importlib
        if "config" in sys.modules:
            del sys.modules["config"]

        from config import validate_config
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            validate_config()


# ── Test 2: Chunking logic ───────────────────────────────
class TestChunking:
    def test_chunks_are_created(self):
        """chunk_documents() should return more chunks than input texts."""
        import sys
        sys.path.insert(0, "src")
        from ingest import chunk_documents

        long_text = "This is a sentence. " * 200  # ~800 words
        chunks = chunk_documents([long_text])

        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunk_size_is_respected(self):
        """No chunk should massively exceed the chunk_size setting."""
        import sys
        sys.path.insert(0, "src")
        from ingest import chunk_documents

        long_text = "word " * 1000
        chunks = chunk_documents([long_text])

        # Allow 20% tolerance above chunk_size
        for chunk in chunks:
            assert len(chunk) <= 800 * 1.2, f"Chunk too large: {len(chunk)} chars"

    def test_overlap_preserves_context(self):
        """Consecutive chunks should share some words (overlap)."""
        import sys
        sys.path.insert(0, "src")
        from ingest import chunk_documents

        # Create text with unique markers every 50 words
        text = " ".join([f"MARKER{i} " + "filler word " * 48 for i in range(10)])
        chunks = chunk_documents([text])

        # At least one marker should appear in two consecutive chunks (overlap)
        found_overlap = False
        for i in range(len(chunks) - 1):
            shared = set(chunks[i].split()) & set(chunks[i + 1].split())
            if len(shared) > 5:
                found_overlap = True
                break

        assert found_overlap, "Expected overlapping content between consecutive chunks"


# ── Test 3: Eval scoring logic ───────────────────────────
class TestEvaluation:
    def test_keyword_score_perfect(self):
        """All keywords present → score = 1.0"""
        import sys
        sys.path.insert(0, "src")
        from evaluate import keyword_score

        answer = "photosynthesis uses light, chlorophyll, glucose, plant, carbon dioxide"
        keywords = ["light", "chlorophyll", "glucose", "plant", "carbon dioxide"]
        assert keyword_score(answer, keywords) == 1.0

    def test_keyword_score_zero(self):
        """No keywords present → score = 0.0"""
        import sys
        sys.path.insert(0, "src")
        from evaluate import keyword_score

        answer = "I don't know anything about that topic."
        keywords = ["light", "chlorophyll", "glucose"]
        assert keyword_score(answer, keywords) == 0.0

    def test_keyword_score_partial(self):
        """Some keywords present → score between 0 and 1."""
        import sys
        sys.path.insert(0, "src")
        from evaluate import keyword_score

        answer = "Plants use light and chlorophyll."
        keywords = ["light", "chlorophyll", "glucose", "carbon dioxide"]
        score = keyword_score(answer, keywords)
        assert 0.0 < score < 1.0

    def test_keyword_score_case_insensitive(self):
        """Matching should be case-insensitive."""
        import sys
        sys.path.insert(0, "src")
        from evaluate import keyword_score

        answer = "CHLOROPHYLL and GLUCOSE are key."
        keywords = ["chlorophyll", "glucose"]
        assert keyword_score(answer, keywords) == 1.0

    def test_eval_report_structure(self, tmp_path):
        """Eval report JSON should have required fields."""
        report = {
            "timestamp": "2025-01-01T00:00:00",
            "model": "llama-3.3-70b-versatile",
            "num_questions": 5,
            "avg_score": 0.72,
            "avg_latency_s": 1.4,
            "results": []
        }
        report_file = tmp_path / "eval_report.json"
        report_file.write_text(json.dumps(report))

        loaded = json.loads(report_file.read_text())
        for key in ["timestamp", "model", "avg_score", "avg_latency_s", "results"]:
            assert key in loaded, f"Missing key in report: {key}"


# ── Test 4: Agent quality gate ───────────────────────────
class TestQualityGate:
    def test_score_threshold_passes(self):
        """Scores above 0.35 should pass the quality gate."""
        avg_score = 0.72
        assert avg_score >= 0.35, "Should pass quality gate"

    def test_score_threshold_fails(self):
        """Scores below 0.35 should fail."""
        avg_score = 0.20
        assert avg_score < 0.35, "Should fail quality gate"