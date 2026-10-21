"""Measure document text-extraction success against expected text in a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from app.services.ingestion import SUPPORTED, load_file
from docx.opc.exceptions import PackageNotFoundError
from lxml.etree import XMLSyntaxError
from pypdf.errors import PdfReadError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "ingestion_cases.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "ingestion.json"


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain a non-empty 'cases' array")

    seen_ids: set[str] = set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Every ingestion case must be a JSON object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every ingestion case must have a non-empty case_id")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(item.get("path"), str) or not item["path"].strip():
            raise ValueError(f"{case_id}: path must be a non-empty string")
        expected_text = item.get("expected_text")
        if (
            not isinstance(expected_text, list)
            or not expected_text
            or any(not isinstance(text, str) or not text.strip() for text in expected_text)
        ):
            raise ValueError(f"{case_id}: expected_text must be a non-empty string array")
    return manifest


def evaluate_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    records = []
    for case in manifest["cases"]:
        source_path = Path(case["path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        extension = source_path.suffix.lower()
        record = {
            "case_id": case["case_id"],
            "file_type": extension.lstrip(".") or "unknown",
            "required_text_count": len(case["expected_text"]),
            "matched_text_count": 0,
            "success": False,
        }

        try:
            if extension not in SUPPORTED:
                raise ValueError(
                    f"Unsupported extension '{extension}'; expected "
                    f"{', '.join(sorted(SUPPORTED))}"
                )
            documents = load_file(source_path)
            extracted_text = _normalize(
                "\n".join(document.page_content for document in documents)
            )
            record["matched_text_count"] = sum(
                _normalize(expected) in extracted_text
                for expected in case["expected_text"]
            )
            record["success"] = (
                record["matched_text_count"] == record["required_text_count"]
            )
            if not record["success"]:
                record["error"] = "expected_text_not_found"
        except (
            BadZipFile,
            OSError,
            PackageNotFoundError,
            PdfReadError,
            RuntimeError,
            ValueError,
            XMLSyntaxError,
        ) as exc:
            record["error"] = type(exc).__name__
        records.append(record)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["file_type"]].append(record)

    total = len(records)
    successful = sum(record["success"] for record in records)
    return {
        "schema_version": 1,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_version": manifest.get("version"),
        "definition": (
            "A file passes only when every non-empty expected_text fragment "
            "is present in the extracted text after whitespace normalization "
            "and case-insensitive matching."
        ),
        "total_files": total,
        "successful_files": successful,
        "parsing_success_rate_pct": round(100 * successful / total, 1),
        "by_file_type": {
            file_type: {
                "total_files": len(type_records),
                "successful_files": sum(record["success"] for record in type_records),
                "success_rate_pct": round(
                    100
                    * sum(record["success"] for record in type_records)
                    / len(type_records),
                    1,
                ),
            }
            for file_type, type_records in sorted(by_type.items())
        },
        "cases": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        report = evaluate_manifest(manifest, args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Ingestion evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Document extraction success: "
        f"{report['successful_files']}/{report['total_files']} "
        f"({report['parsing_success_rate_pct']:.1f}%)"
    )
    for file_type, metrics in report["by_file_type"].items():
        print(
            f"  .{file_type}: {metrics['successful_files']}/"
            f"{metrics['total_files']} ({metrics['success_rate_pct']:.1f}%)"
        )
    print(f"Report saved to {args.output.resolve()}")
    failed_count = report["total_files"] - report["successful_files"]
    if failed_count:
        print(f"Review {failed_count} failed case(s) in the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
