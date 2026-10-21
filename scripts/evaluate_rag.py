"""Run and score a small paired evaluation of FixPilot's RAG workflows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation" / "questions.json"
DEFAULT_RESULTS = ROOT / "evaluation" / "results" / "latest.json"
SYSTEM_NAMES = ("basic_rag", "agentic_rag")
EXPECTED_SOURCES = {"private_kb", "web_search"}


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    questions = dataset.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must contain a non-empty 'questions' array")

    seen_ids: set[str] = set()
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Every question must be a JSON object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every question must have a non-empty case_id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"{case_id}: question must be a non-empty string")
        if item.get("expected_source") not in EXPECTED_SOURCES:
            raise ValueError(
                f"{case_id}: expected_source must be one of "
                f"{', '.join(sorted(EXPECTED_SOURCES))}"
            )
        if not isinstance(item.get("expected_answer_points"), list):
            raise ValueError(f"{case_id}: expected_answer_points must be an array")
    return dataset


def run_evaluation(dataset: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    from app.rag.workflow import ask, llm
    from app.rag.vectorstore import get_retriever

    retriever = get_retriever()
    model = llm()
    cases = []

    for question_case in dataset["questions"]:
        question = question_case["question"]
        started = time.perf_counter()
        documents = retriever.invoke(question)
        context = "\n\n".join(
            f"[Source: {document.metadata.get('source', 'unknown')}]\n"
            f"{document.page_content}"
            for document in documents
        )

        baseline_response = model.invoke(
            "You are an IT support assistant. Answer using only the retrieved "
            "company knowledge below. If it does not contain the answer, say "
            "that the evidence is insufficient; do not invent company policy.\n"
            f"Question: {question}\n\nRetrieved company knowledge:\n{context}"
        )
        baseline_duration_ms = round((time.perf_counter() - started) * 1000, 1)
        baseline_result = {
            "source_used": "private_kb",
            "selected_source": "private_kb",
            "web_search_used": False,
            "answer": baseline_response.content,
            "trace": ["Retrieve top-k private KB chunks", "Generate answer without evidence grading"],
            "citations": [
                document.metadata.get("source", "unknown")
                for document in documents
            ],
            "duration_ms": baseline_duration_ms,
            "answer_correct": None,
            "unsupported_claims": None,
        }

        started = time.perf_counter()
        agent_response = ask(question)
        agent_duration_ms = round((time.perf_counter() - started) * 1000, 1)
        trace = agent_response.get("trace", [])
        selected_source = next(
            (
                source
                for trace_marker, source in (
                    ("Answer generation → PRIVATE KB", "private_kb"),
                    ("Answer generation → WEB SEARCH", "web_search"),
                )
                if trace_marker in trace
            ),
            "web_search"
            if "Web fallback → Tavily search" in trace
            else agent_response.get("source_used", "unknown"),
        )
        agent_result = {
            "source_used": agent_response.get("source_used", "unknown"),
            "selected_source": selected_source,
            "web_search_used": "Web fallback → Tavily search" in trace,
            "answer": agent_response.get("answer", ""),
            "trace": trace,
            "citations": agent_response.get("citations", []),
            "duration_ms": agent_duration_ms,
            "answer_correct": None,
            "unsupported_claims": None,
        }

        cases.append(
            {
                "case_id": question_case["case_id"],
                "question": question,
                "expected_source": question_case["expected_source"],
                "expected_answer_points": question_case["expected_answer_points"],
                "basic_rag": baseline_result,
                "agentic_rag": agent_result,
            }
        )
        print(
            f"Completed {len(cases)}/{len(dataset['questions'])}: "
            f"{question_case['case_id']}"
        )

    return {
        "schema_version": 1,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path.resolve()),
        "dataset_version": dataset.get("version"),
        "cases": cases,
        "annotation_guide": {
            "answer_correct": "Set to true only when the answer covers the material expected answer points accurately; otherwise false.",
            "unsupported_claims": "Set to true if the answer makes at least one factual claim unsupported by its cited/retrieved evidence; otherwise false. Judge refusals as false if they make no unsupported factual claims.",
        },
    }


def _percentage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(100 * numerator / denominator, 1)


def _score_system(cases: list[dict[str, Any]], system_name: str) -> dict[str, Any]:
    route_correct = 0
    fallback_true_positive = 0
    fallback_false_positive = 0
    fallback_false_negative = 0
    answer_annotations: list[bool] = []
    unsupported_annotations: list[bool] = []
    durations: list[float] = []
    completed = 0

    for case in cases:
        result = case.get(system_name)
        if not isinstance(result, dict):
            continue
        actual_source = result.get("selected_source", result.get("source_used"))
        expected_source = case.get("expected_source")
        if actual_source:
            completed += 1
            route_correct += actual_source == expected_source
            expected_fallback = expected_source == "web_search"
            actual_fallback = result.get(
                "web_search_used", actual_source == "web_search"
            )
            if not isinstance(actual_fallback, bool):
                raise ValueError(
                    f"{case.get('case_id', 'unknown')}/{system_name}: "
                    "web_search_used must be a boolean"
                )
            fallback_true_positive += expected_fallback and actual_fallback
            fallback_false_positive += not expected_fallback and actual_fallback
            fallback_false_negative += expected_fallback and not actual_fallback
        duration = result.get("duration_ms")
        if isinstance(duration, (int, float)) and duration >= 0:
            durations.append(float(duration))

        for field, annotations in (
            ("answer_correct", answer_annotations),
            ("unsupported_claims", unsupported_annotations),
        ):
            value = result.get(field)
            if value is not None:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"{case.get('case_id', 'unknown')}/{system_name}: "
                        f"{field} must be true, false, or null"
                    )
                annotations.append(value)

    fallback_precision_denominator = fallback_true_positive + fallback_false_positive
    fallback_recall_denominator = fallback_true_positive + fallback_false_negative
    return {
        "completed_cases": completed,
        "source_selection_accuracy_pct": _percentage(route_correct, completed),
        "web_fallback_precision_pct": _percentage(
            fallback_true_positive, fallback_precision_denominator
        ),
        "web_fallback_recall_pct": _percentage(
            fallback_true_positive, fallback_recall_denominator
        ),
        "answer_accuracy_pct": _percentage(
            sum(answer_annotations), len(answer_annotations)
        ),
        "answer_annotations": len(answer_annotations),
        "unsupported_claim_rate_pct": _percentage(
            sum(unsupported_annotations), len(unsupported_annotations)
        ),
        "unsupported_claim_annotations": len(unsupported_annotations),
        "mean_latency_ms": (
            round(sum(durations) / len(durations), 1) if durations else None
        ),
    }


def score_results(results: dict[str, Any]) -> dict[str, Any]:
    cases = results.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Results must contain a non-empty 'cases' array")
    systems = {
        system_name: _score_system(cases, system_name)
        for system_name in SYSTEM_NAMES
    }
    baseline = systems["basic_rag"]
    agentic = systems["agentic_rag"]
    baseline_unsupported = baseline["unsupported_claim_rate_pct"]
    agentic_unsupported = agentic["unsupported_claim_rate_pct"]
    unsupported_claim_rate_reduction = None
    if baseline_unsupported not in (None, 0) and agentic_unsupported is not None:
        unsupported_claim_rate_reduction = round(
            100 * (baseline_unsupported - agentic_unsupported) / baseline_unsupported,
            1,
        )
    return {
        "schema_version": 1,
        "run_at": results.get("run_at"),
        "case_count": len(cases),
        "systems": systems,
        "agentic_vs_basic_rag": {
            "source_selection_accuracy_delta_percentage_points": _percentage_delta(
                agentic["source_selection_accuracy_pct"],
                baseline["source_selection_accuracy_pct"],
            ),
            "answer_accuracy_delta_percentage_points": _percentage_delta(
                agentic["answer_accuracy_pct"], baseline["answer_accuracy_pct"]
            ),
            "unsupported_claim_rate_reduction_pct": unsupported_claim_rate_reduction,
            "mean_latency_change_pct": _relative_change(
                baseline["mean_latency_ms"], agentic["mean_latency_ms"]
            ),
        },
        "metric_notes": {
            "answer_accuracy_pct": "Calculated only from manually annotated answers; null means none are annotated.",
            "unsupported_claim_rate_pct": "Calculated only from manually annotated answers; this is a human-reviewed proxy, not an automatic hallucination detector.",
            "source_selection_accuracy_pct": "Compared with each case's expected_source label.",
            "web_fallback_precision_pct": "Of answers sent to web search, the fraction whose expected source was web search.",
            "web_fallback_recall_pct": "Of cases labeled for web search, the fraction actually sent to web search.",
            "mean_latency_ms": "Mean elapsed time for completed system calls in this run; excludes benchmark setup and manual review.",
        },
    }


def _percentage_delta(new_value: float | None, old_value: float | None) -> float | None:
    if new_value is None or old_value is None:
        return None
    return round(new_value - old_value, 1)


def _relative_change(old_value: float | None, new_value: float | None) -> float | None:
    if old_value in (None, 0) or new_value is None:
        return None
    return round(100 * (new_value - old_value) / old_value, 1)


def _print_report(report: dict[str, Any]) -> None:
    print(f"Evaluation summary ({report['case_count']} cases)")
    for system_name, metrics in report["systems"].items():
        print(f"\n{system_name}")
        print(
            "  Source-selection accuracy: "
            f"{_display_percentage(metrics['source_selection_accuracy_pct'])}"
        )
        print(
            "  Web fallback precision / recall: "
            f"{_display_percentage(metrics['web_fallback_precision_pct'])} / "
            f"{_display_percentage(metrics['web_fallback_recall_pct'])}"
        )
        print(
            "  Human-scored answer accuracy: "
            f"{_display_percentage(metrics['answer_accuracy_pct'])} "
            f"({metrics['answer_annotations']} annotated)"
        )
        print(
            "  Human-scored unsupported-claim rate: "
            f"{_display_percentage(metrics['unsupported_claim_rate_pct'])} "
            f"({metrics['unsupported_claim_annotations']} annotated)"
        )
        print(f"  Mean latency: {_display_latency(metrics['mean_latency_ms'])}")

    comparison = report["agentic_vs_basic_rag"]
    print("\nAgentic RAG vs. basic RAG")
    print(
        "  Answer-accuracy change: "
        f"{_display_delta(comparison['answer_accuracy_delta_percentage_points'])}"
    )
    print(
        "  Unsupported-claim-rate reduction: "
        f"{_display_percentage(comparison['unsupported_claim_rate_reduction_pct'])}"
    )
    print(
        "  Mean-latency change: "
        f"{_display_delta(comparison['mean_latency_change_pct'], '%')}"
    )

    if any(
        metrics["answer_annotations"] == 0
        or metrics["unsupported_claim_annotations"] == 0
        for metrics in report["systems"].values()
    ):
        print(
            "\nAnswer quality metrics are incomplete. Review each answer in the "
            "results JSON, fill in answer_correct and unsupported_claims, then "
            "run the score command again."
        )


def _display_percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def _display_latency(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f} ms"


def _display_delta(value: float | None, suffix: str = " percentage points") -> str:
    return "N/A" if value is None else f"{value:+.1f}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Run both RAG approaches on the dataset")
    run_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)

    score_parser = commands.add_parser(
        "score", help="Calculate metrics from results and manual annotations"
    )
    score_parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    score_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS.with_name("report.json"),
    )

    args = parser.parse_args()
    try:
        if args.command == "run":
            dataset = load_dataset(args.dataset)
            results = run_evaluation(dataset, args.dataset)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Raw results saved to {args.output.resolve()}")
            print(
                "Review and annotate both answers for every case before treating "
                "answer-quality metrics as complete."
            )
        else:
            with args.input.open(encoding="utf-8") as input_file:
                results = json.load(input_file)
            report = score_results(results)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _print_report(report)
            print(f"\nScored report saved to {args.output.resolve()}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
