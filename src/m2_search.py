from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import math
import os, sys
import re
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    MOCK_MODE)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        # Keep underscores: the same tokenizer is applied to documents and queries,
        # so Vietnamese compound words become consistent BM25 terms.
        return word_tokenize(text, format="text").casefold()
    except (ImportError, ModuleNotFoundError):
        # Dependency-free fallback for minimal/offline environments.
        return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c.get("text", "")).split() for c in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except (ImportError, ModuleNotFoundError):
            self.bm25 = _FallbackBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        query_tokens = segment_vietnamese(query).split()
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results = []
        for index in indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(SearchResult(document.get("text", ""), score,
                                        dict(document.get("metadata", {})), "bm25"))
            if len(results) >= top_k:
                break
        return results


class _FallbackBM25:
    """Small BM25Okapi-compatible fallback used only when rank-bm25 is absent."""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.term_frequencies = [Counter(document) for document in corpus]
        document_frequency = Counter(term for document in corpus for term in set(document))
        count = len(corpus)
        self.idf = {term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
                    for term, freq in document_frequency.items()}

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for document, frequencies in zip(self.corpus, self.term_frequencies):
            score = 0.0
            length_norm = 1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9)
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                    )
            scores.append(score)
        return scores


class DenseSearch:
    def __init__(self):
        self._client = None
        self._encoder = None
        self._mock_documents: list[dict] = []

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return self._client

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if MOCK_MODE:
            self._mock_documents = list(chunks)
            return
        from qdrant_client.models import Distance, PointStruct, VectorParams
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return
        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True,
                                             normalize_embeddings=True)
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip() or top_k <= 0:
            return []
        if MOCK_MODE:
            query_vector = _mock_embedding(query)
            scored = [(_cosine(query_vector, _mock_embedding(doc.get("text", ""))), doc)
                      for doc in self._mock_documents]
            scored.sort(key=lambda item: item[0], reverse=True)
            return [SearchResult(doc.get("text", ""), score,
                                 dict(doc.get("metadata", {})), "dense")
                    for score, doc in scored[:top_k] if score > 0]
        vector = self._get_encoder().encode(query, normalize_embeddings=True)
        query_vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        response = self.client.query_points(
            collection_name=collection, query=query_vector, limit=top_k,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            results.append(SearchResult(text, float(point.score), payload, "dense"))
        return results


def _mock_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Deterministic hashed vector for mock-mode control-flow testing."""
    import hashlib
    vector = [0.0] * dimensions
    for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1.0
    return vector


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    fused: dict[str, dict] = {}
    for ranked_results in results_list:
        seen_in_list: set[str] = set()
        for rank, result in enumerate(ranked_results):
            if result.text in seen_in_list:
                continue
            seen_in_list.add(result.text)
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [SearchResult(item["result"].text, item["score"],
                         dict(item["result"].metadata), "hybrid") for item in ranked]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
