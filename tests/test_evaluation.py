import unittest
import tempfile
from pathlib import Path

from scripts.evaluate_ingestion import evaluate_manifest
from scripts.evaluate_rag import load_dataset, score_results


class EvaluationMetricsTests(unittest.TestCase):
    def test_ingestion_success_requires_all_expected_text(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory)
            sample_path = manifest_dir / "handbook.md"
            sample_path.write_text("VPN Access\nUse MFA to sign in.", encoding="utf-8")
            manifest_path = manifest_dir / "manifest.json"
            manifest = {
                "version": 1,
                "cases": [
                    {
                        "case_id": "complete",
                        "path": "handbook.md",
                        "expected_text": ["VPN Access", "Use MFA"],
                    },
                    {
                        "case_id": "missing-text",
                        "path": "handbook.md",
                        "expected_text": ["VPN Access", "Missing policy text"],
                    },
                ],
            }

            report = evaluate_manifest(manifest, manifest_path)

        self.assertEqual(report["total_files"], 2)
        self.assertEqual(report["successful_files"], 1)
        self.assertEqual(report["parsing_success_rate_pct"], 50.0)
        self.assertEqual(report["by_file_type"]["md"]["success_rate_pct"], 50.0)

    def test_scores_source_fallback_answers_and_latency(self):
        results = {
            "run_at": "2026-10-21T10:00:00+00:00",
            "cases": [
                {
                    "case_id": "private",
                    "expected_source": "private_kb",
                    "basic_rag": {
                        "source_used": "private_kb",
                        "duration_ms": 100,
                        "answer_correct": True,
                        "unsupported_claims": False,
                    },
                    "agentic_rag": {
                        "source_used": "private_kb",
                        "duration_ms": 300,
                        "answer_correct": True,
                        "unsupported_claims": False,
                    },
                },
                {
                    "case_id": "web",
                    "expected_source": "web_search",
                    "basic_rag": {
                        "source_used": "private_kb",
                        "duration_ms": 100,
                        "answer_correct": False,
                        "unsupported_claims": True,
                    },
                    "agentic_rag": {
                        "source_used": "web_search",
                        "duration_ms": 500,
                        "answer_correct": True,
                        "unsupported_claims": False,
                    },
                },
            ],
        }

        report = score_results(results)

        baseline = report["systems"]["basic_rag"]
        agentic = report["systems"]["agentic_rag"]
        self.assertEqual(baseline["source_selection_accuracy_pct"], 50.0)
        self.assertEqual(baseline["web_fallback_precision_pct"], None)
        self.assertEqual(baseline["web_fallback_recall_pct"], 0.0)
        self.assertEqual(baseline["answer_accuracy_pct"], 50.0)
        self.assertEqual(baseline["unsupported_claim_rate_pct"], 50.0)
        self.assertEqual(baseline["mean_latency_ms"], 100.0)
        self.assertEqual(agentic["source_selection_accuracy_pct"], 100.0)
        self.assertEqual(agentic["web_fallback_precision_pct"], 100.0)
        self.assertEqual(agentic["web_fallback_recall_pct"], 100.0)
        self.assertEqual(agentic["answer_accuracy_pct"], 100.0)
        self.assertEqual(agentic["unsupported_claim_rate_pct"], 0.0)
        self.assertEqual(agentic["mean_latency_ms"], 400.0)
        self.assertEqual(
            report["agentic_vs_basic_rag"][
                "unsupported_claim_rate_reduction_pct"
            ],
            100.0,
        )
        self.assertEqual(
            report["agentic_vs_basic_rag"]["answer_accuracy_delta_percentage_points"],
            50.0,
        )

    def test_unannotated_answers_remain_unscored(self):
        report = score_results(
            {
                "cases": [
                    {
                        "case_id": "pending",
                        "expected_source": "private_kb",
                        "agentic_rag": {
                            "source_used": "private_kb",
                            "answer_correct": None,
                            "unsupported_claims": None,
                        },
                    }
                ]
            }
        )
        metrics = report["systems"]["agentic_rag"]
        self.assertIsNone(metrics["answer_accuracy_pct"])
        self.assertIsNone(metrics["unsupported_claim_rate_pct"])
        self.assertEqual(metrics["answer_annotations"], 0)

    def test_rejects_non_boolean_manual_annotations(self):
        with self.assertRaisesRegex(ValueError, "answer_correct must be"):
            score_results(
                {
                    "cases": [
                        {
                            "case_id": "bad-annotation",
                            "expected_source": "private_kb",
                            "agentic_rag": {
                                "source_used": "private_kb",
                                "answer_correct": "yes",
                            },
                        }
                    ]
                }
            )

    def test_loads_checked_in_benchmark(self):
        dataset = load_dataset(
            Path(__file__).resolve().parents[1] / "evaluation" / "questions.json"
        )
        self.assertEqual(len(dataset["questions"]), 11)
        self.assertEqual(
            sum(case["expected_source"] == "web_search" for case in dataset["questions"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
