from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
import hashlib
from functools import lru_cache
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  ⚠️  Bỏ qua {os.path.basename(path)}: chưa cài pypdf.")
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n+", text.strip()) if s.strip()]
    if not sentences:
        return []

    embeddings = _sentence_embeddings(sentences)
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if _cosine_similarity(embeddings[i - 1], embeddings[i]) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": i})
            for i, group in enumerate(groups)]


@lru_cache(maxsize=1)
def _semantic_model():
    """Load the sentence embedding model only once."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _sentence_embeddings(sentences: list[str]):
    try:
        return _semantic_model().encode(sentences, normalize_embeddings=True)
    except Exception:
        # Offline fallback: hashed bag-of-words vectors still provide cosine-based
        # boundaries when sentence-transformers or the model is unavailable.
        vectors = []
        for sentence in sentences:
            vector = [0.0] * 384
            for token in re.findall(r"\w+", sentence.casefold(), flags=re.UNICODE):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
                vector[int.from_bytes(digest, "big") % len(vector)] += 1.0
            vectors.append(vector)
        return vectors


def _cosine_similarity(a, b) -> float:
    dot_product = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = sum(float(x) ** 2 for x in a) ** 0.5
    norm_b = sum(float(y) ** 2 for y in b) ** 0.5
    return dot_product / (norm_a * norm_b + 1e-9)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if not text.strip():
        return [], []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    parent_texts = _pack_blocks(paragraphs, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    document_key = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{document_key}_{parent_index}"
        parent_meta = {**metadata, "chunk_type": "parent", "chunk_index": parent_index,
                       "parent_id": parent_id}
        parents.append(Chunk(parent_text, parent_meta, parent_id))

        for child_text in _pack_blocks(re.split(r"\n\s*\n+", parent_text), child_size):
            child_meta = {**metadata, "chunk_type": "child", "chunk_index": len(children),
                          "parent_id": parent_id}
            children.append(Chunk(child_text, child_meta, parent_id))
    return parents, children


def _pack_blocks(blocks: list[str], max_size: int) -> list[str]:
    """Pack blocks up to max_size and safely split a single oversized block."""
    normalized: list[str] = []
    for block in blocks:
        block = block.strip()
        while len(block) > max_size:
            split_at = block.rfind(" ", 0, max_size + 1)
            split_at = split_at if split_at > 0 else max_size
            normalized.append(block[:split_at].strip())
            block = block[split_at:].strip()
        if block:
            normalized.append(block)

    packed: list[str] = []
    current = ""
    for block in normalized:
        candidate = f"{current}\n\n{block}" if current else block
        if current and len(candidate) > max_size:
            packed.append(current)
            current = block
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    chunks: list[Chunk] = []
    preamble_end = matches[0].start() if matches else len(text)
    preamble = text[:preamble_end].strip()
    if preamble:
        chunks.append(Chunk(preamble, {**metadata, "section": "preamble",
                                      "strategy": "structure", "chunk_index": 0}))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(Chunk(
            text[match.start():end].strip(),
            {**metadata, "section": match.group(2).strip(),
             "heading_level": len(match.group(1)), "strategy": "structure",
             "chunk_index": len(chunks)},
        ))
    if not matches and not chunks:
        chunks.append(Chunk(text.strip(), {**metadata, "section": "document",
                                          "strategy": "structure", "chunk_index": 0}))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
