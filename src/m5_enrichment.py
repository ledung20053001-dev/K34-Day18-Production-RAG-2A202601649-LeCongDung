from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, sys
import re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MOCK_MODE, OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    result = _enrich_single_call(text, "")
    if result.get("summary"):
        return result["summary"]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    result = _enrich_single_call(text, "")
    questions = result.get("questions", [])
    if questions:
        return questions[:n_questions]
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    return [f"{sentence.rstrip('.')}?" for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _enrich_single_call(text, document_title).get("context", "")
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    return _enrich_single_call(text, "").get("metadata", {})


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if MOCK_MODE or not OPENAI_API_KEY or not text.strip():
        return {}

    system_prompt = """Bạn làm giàu chunk cho hệ thống RAG tiếng Việt.
Chỉ trả về một JSON object hợp lệ, không Markdown, theo schema:
{
  "summary": "Tóm tắt chính xác nội dung chunk trong 1-2 câu",
  "questions": ["2-3 câu hỏi mà chunk có thể trả lời"],
  "context": "Một câu nêu chunk thuộc phần/ngữ cảnh nào của tài liệu",
  "metadata": {
    "topic": "chủ đề chính",
    "keywords": ["từ khóa chính"],
    "entities": ["thực thể quan trọng"],
    "category": "policy|hr|it|finance|safety|compliance|general",
    "language": "vi|en"
  }
}
Không suy diễn thông tin không xuất hiện trong chunk hoặc ngữ cảnh tài liệu."""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Ngữ cảnh tài liệu: {source or 'Không cung cấp'}\n\nChunk:\n{text}"},
            ],
            temperature=0,
            max_tokens=400,
        )
        content = response.choices[0].message.content or ""
        return _normalize_enrichment(_parse_json_object(content))
    except Exception as error:
        print(f"  ⚠️  Enrichment API failed; dùng chunk gốc: {error}")
        return {}


def _parse_json_object(content: str) -> dict:
    """Parse JSON and tolerate an accidental Markdown code fence."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Enrichment response must be a JSON object")
    return value


def _normalize_enrichment(value: dict) -> dict:
    """Validate LLM-shaped data before it reaches indexing metadata."""
    summary = value.get("summary", "")
    context = value.get("context", value.get("contextual_info", ""))
    questions = value.get("questions", value.get("hypothetical_questions", []))
    metadata = value.get("metadata", value.get("keywords_metadata", {}))
    return {
        "summary": summary.strip() if isinstance(summary, str) else "",
        "questions": [q.strip() for q in questions[:3] if isinstance(q, str) and q.strip()]
        if isinstance(questions, list) else [],
        "context": context.strip() if isinstance(context, str) else "",
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
