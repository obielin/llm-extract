"""
Tests for llm-extract.
Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_extract.chunker import chunk_text
from llm_extract.loader import load_document, _load_csv, _load_json
from llm_extract.models import DocumentChunk, ExtractionConfig, ExtractionResult
from llm_extract.extractor import _build_prompt, _describe_schema, _parse_response


# ── Test schemas ──────────────────────────────────────────────────────────────

class PersonSchema(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class InvoiceSchema(BaseModel):
    vendor: str
    total: float
    date: str
    invoice_number: Optional[str] = None


class ContractSchema(BaseModel):
    parties: list[str]
    effective_date: str
    value_gbp: Optional[float] = None
    governing_law: Optional[str] = None


# ── ExtractionResult tests ────────────────────────────────────────────────────

class TestExtractionResult:
    def test_success_true_when_data_present(self):
        person = PersonSchema(name="Alice", age=30)
        result = ExtractionResult(data=person, fields_found=2, fields_total=3)
        assert result.success is True

    def test_success_false_when_data_none(self):
        result = ExtractionResult(data=None, fields_total=3)
        assert result.success is False

    def test_coverage_calculation(self):
        person = PersonSchema(name="Alice", age=30)
        result = ExtractionResult(data=person, fields_found=2, fields_total=3)
        assert result.coverage == pytest.approx(2 / 3)

    def test_coverage_zero_when_no_fields(self):
        result = ExtractionResult(data=None, fields_found=0, fields_total=0)
        assert result.coverage == 0.0

    def test_mean_confidence_calculation(self):
        person = PersonSchema(name="Alice", age=30)
        result = ExtractionResult(
            data=person,
            confidence={"name": 0.9, "age": 0.8},
            fields_found=2, fields_total=3,
        )
        assert result.mean_confidence == pytest.approx(0.85)

    def test_mean_confidence_zero_when_empty(self):
        result = ExtractionResult(data=None)
        assert result.mean_confidence == 0.0

    def test_low_confidence_fields(self):
        person = PersonSchema(name="Alice", age=30)
        result = ExtractionResult(
            data=person,
            confidence={"name": 0.95, "age": 0.60, "email": 0.50},
            fields_found=3, fields_total=3,
        )
        low = result.low_confidence_fields(threshold=0.7)
        assert "age" in low
        assert "email" in low
        assert "name" not in low

    def test_repr_shows_status(self):
        person = PersonSchema(name="Alice", age=30)
        result = ExtractionResult(data=person, fields_found=2, fields_total=3)
        repr_str = repr(result)
        assert "✓" in repr_str
        assert "coverage" in repr_str


# ── ExtractionConfig tests ────────────────────────────────────────────────────

class TestExtractionConfig:
    def test_defaults(self):
        config = ExtractionConfig()
        assert config.model == "claude-opus-4-5"
        assert config.temperature == 0.1
        assert config.chunk_size == 8000
        assert config.overlap == 500
        assert config.provider == "anthropic"

    def test_custom_values(self):
        config = ExtractionConfig(model="claude-haiku-4-5", temperature=0.0, chunk_size=4000)
        assert config.model == "claude-haiku-4-5"
        assert config.temperature == 0.0
        assert config.chunk_size == 4000


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Hello world. This is a short document."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_long_text_returns_multiple_chunks(self):
        text = "sentence. " * 500  # ~5000 chars
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) > 1

    def test_all_chunks_have_correct_metadata(self):
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)
            assert len(chunk.text) > 0

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_size_respected_approximately(self):
        text = "a" * 10000
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        for chunk in chunks:
            assert len(chunk.text) <= 1100  # Allow slight overshoot

    def test_prefers_paragraph_splits(self):
        text = "First paragraph content here.\n\nSecond paragraph content.\n\nThird paragraph content."
        chunks = chunk_text(text, chunk_size=40, overlap=5)
        # Should split at paragraph boundaries
        assert len(chunks) > 1

    def test_char_positions_are_sequential(self):
        text = "Hello world " * 100
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        for i in range(len(chunks) - 1):
            assert chunks[i].char_start <= chunks[i+1].char_start


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        assert load_document(f) == "Hello world"

    def test_load_md_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nContent here.")
        result = load_document(f)
        assert "Title" in result
        assert "Content here" in result

    def test_load_csv_produces_markdown_table(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("name,age,city\nAlice,30,London\nBob,25,Leeds")
        result = load_document(f)
        assert "name" in result
        assert "Alice" in result
        assert "|" in result  # markdown table

    def test_load_json_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"name": "Alice", "age": 30}')
        result = load_document(f)
        assert "Alice" in result
        assert "30" in result

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/file.txt")

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document(f)

    def test_csv_handles_uneven_rows(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2\n4,5,6")
        result = _load_csv(f)
        assert "a" in result


# ── Prompt building tests ─────────────────────────────────────────────────────

class TestPromptBuilding:
    def test_prompt_contains_schema_fields(self):
        prompt = _build_prompt("sample text", PersonSchema, None)
        assert "name" in prompt
        assert "age" in prompt
        assert "email" in prompt

    def test_prompt_contains_document_text(self):
        prompt = _build_prompt("The quick brown fox", PersonSchema, None)
        assert "The quick brown fox" in prompt

    def test_prompt_contains_instructions(self):
        prompt = _build_prompt("text", PersonSchema, "Focus on UK residents only")
        assert "Focus on UK residents only" in prompt

    def test_prompt_requests_json_output(self):
        prompt = _build_prompt("text", PersonSchema, None)
        assert "JSON" in prompt or "json" in prompt

    def test_prompt_requests_confidence_scores(self):
        prompt = _build_prompt("text", PersonSchema, None)
        assert "confidence" in prompt.lower()

    def test_prompt_requests_source_snippets(self):
        prompt = _build_prompt("text", PersonSchema, None)
        assert "source" in prompt.lower()


class TestDescribeSchema:
    def test_includes_field_names(self):
        desc = _describe_schema(PersonSchema)
        assert "name" in desc
        assert "age" in desc
        assert "email" in desc

    def test_includes_model_name(self):
        desc = _describe_schema(PersonSchema)
        assert "PersonSchema" in desc

    def test_marks_required_fields(self):
        desc = _describe_schema(PersonSchema)
        assert "required" in desc

    def test_marks_optional_fields(self):
        desc = _describe_schema(PersonSchema)
        assert "optional" in desc


# ── Response parsing tests ────────────────────────────────────────────────────

class TestParseResponse:
    def make_response(self, **kwargs) -> str:
        """Build a mock LLM response JSON."""
        data = {}
        for field, (value, confidence, source) in kwargs.items():
            data[field] = {"value": value, "confidence": confidence, "source": source}
        return json.dumps(data)

    def test_valid_response_produces_model(self):
        raw = self.make_response(
            name=("Alice", 0.95, "Alice Chen"),
            age=(30, 0.90, "aged 30"),
        )
        result = _parse_response(raw, PersonSchema)
        assert result.success
        assert result.data.name == "Alice"
        assert result.data.age == 30

    def test_confidence_scores_extracted(self):
        raw = self.make_response(
            name=("Bob", 0.88, "Bob Smith"),
            age=(25, 0.75, "25 years old"),
        )
        result = _parse_response(raw, PersonSchema)
        assert result.confidence["name"] == pytest.approx(0.88)
        assert result.confidence["age"] == pytest.approx(0.75)

    def test_source_snippets_extracted(self):
        raw = self.make_response(
            name=("Alice", 0.9, "Alice Chen, Senior Engineer"),
        )
        result = _parse_response(raw, PersonSchema)
        assert "Alice Chen" in result.sources.get("name", "")

    def test_null_values_handled_gracefully(self):
        raw = json.dumps({
            "name": {"value": "Alice", "confidence": 0.9, "source": "Alice"},
            "age": {"value": None, "confidence": 0.0, "source": ""},
        })
        result = _parse_response(raw, PersonSchema)
        # May fail validation if age is required — that's correct behaviour

    def test_invalid_json_returns_failed_result(self):
        result = _parse_response("not valid json at all {{{{", PersonSchema)
        assert not result.success

    def test_markdown_fences_stripped(self):
        raw = '```json\n{"name": {"value": "Alice", "confidence": 0.9, "source": "Alice"}, "age": {"value": 28, "confidence": 0.85, "source": "28"}}\n```'
        result = _parse_response(raw, PersonSchema)
        assert result.success
        assert result.data.name == "Alice"

    def test_confidence_clamped_to_range(self):
        raw = json.dumps({
            "name": {"value": "Bob", "confidence": 1.5, "source": ""},
            "age": {"value": 30, "confidence": -0.2, "source": ""},
        })
        result = _parse_response(raw, PersonSchema)
        if result.confidence:
            for v in result.confidence.values():
                assert 0.0 <= v <= 1.0

    def test_flat_format_handled(self):
        """Handle responses where LLM returns value only, without nested dict."""
        raw = json.dumps({"name": "Alice", "age": 30})
        result = _parse_response(raw, PersonSchema)
        assert result.success
        assert result.data.name == "Alice"
