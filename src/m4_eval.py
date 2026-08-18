from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MOCK_MODE, OPENAI_API_KEY, TEST_SET_PATH


METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    fallback = _fallback_results(questions, answers, contexts, ground_truths)
    if not questions:
        return fallback
    if MOCK_MODE:
        print("  ⚠️  RAG mock mode; bỏ qua RAGAS và trả về điểm 0.")
        return fallback
    if not OPENAI_API_KEY:
        print("  ⚠️  Không có OPENAI_API_KEY; bỏ qua RAGAS và trả về điểm 0.")
        return fallback

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        evaluation = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        frame = evaluation.to_pandas()
        per_question = []
        for _, row in frame.iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                **{name: _safe_score(row.get(name, 0.0)) for name in METRIC_NAMES},
            ))
        aggregates = {
            name: (sum(getattr(item, name) for item in per_question) / len(per_question)
                   if per_question else 0.0)
            for name in METRIC_NAMES
        }
        return {**aggregates, "per_question": per_question}
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed; trả về điểm 0: {error}")
        return fallback


def _safe_score(value) -> float:
    try:
        score = float(value)
        return score if math.isfinite(score) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fallback_results(questions, answers, contexts, ground_truths) -> dict:
    per_question = [EvalResult(q, a, list(c), gt, 0.0, 0.0, 0.0, 0.0)
                    for q, a, c, gt in zip(questions, answers, contexts, ground_truths)]
    return {**{name: 0.0 for name in METRIC_NAMES}, "per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10,
                     threshold: float = 0.7) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "context_recall": (
            "retrieval",
            "Thiếu chunk chứa thông tin cần thiết (retrieval miss).",
            "Điều chỉnh kích thước chunk; bổ sung BM25/query expansion và tăng candidate top-k.",
        ),
        "context_precision": (
            "retrieval",
            "Các chunk không liên quan được xếp quá cao hoặc reranker lọc nhầm.",
            "Tối ưu hybrid weights/RRF; thêm metadata filter và kiểm tra CrossEncoder reranking.",
        ),
        "faithfulness": (
            "generation",
            "Câu trả lời chứa thông tin không được context hỗ trợ (hallucination).",
            "Siết prompt chỉ trả lời từ context, giảm temperature và yêu cầu trích dẫn bằng chứng.",
        ),
        "answer_relevancy": (
            "generation",
            "Câu trả lời không đúng trọng tâm hoặc diễn giải thừa.",
            "Cải thiện prompt theo câu hỏi, yêu cầu câu trả lời trực tiếp và giới hạn độ dài.",
        ),
    }

    ranked = []
    for result in eval_results:
        scores = {name: _safe_score(getattr(result, name)) for name in METRIC_NAMES}
        average = sum(scores.values()) / len(scores)
        if average >= threshold:
            continue
        worst_metric = min(scores, key=scores.get)
        failure_type, diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append((average, {
            "question": result.question,
            "expected": result.ground_truth,
            "got": result.answer,
            "contexts": result.contexts,
            "failure_type": failure_type,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "average_score": average,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        }))
    ranked.sort(key=lambda item: item[0])
    return [failure for _, failure in ranked[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
