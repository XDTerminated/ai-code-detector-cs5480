"""Pretty-print every metric/error JSON under ``reports/metrics/``.

Useful as a quick post-training summary:

    uv run python scripts/show_results.py

Optionally pass ``--update-baseline-md`` to splice the baseline test metrics
into the placeholder table inside ``reports/results_baseline.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.training.metrics import ClassificationMetrics  # noqa: E402

logger = logging.getLogger("show_results")


def _print_metrics(name: str, payload: dict) -> None:
    metrics = ClassificationMetrics.from_dict(payload)
    print(f"\n=== {name} ===")
    print(metrics.pretty())


def _print_error_summary(name: str, payload: dict) -> None:
    print(f"\n=== {name} ===")
    print(json.dumps(payload, indent=2))


def _build_baseline_test_table(payload: dict) -> str:
    metrics = ClassificationMetrics.from_dict(payload)
    human = metrics.per_class.get("human", {})
    ai = metrics.per_class.get("ai", {})
    rows = [
        ("Accuracy", metrics.accuracy),
        ("Macro F1", metrics.macro_f1),
        ("AUC-ROC", metrics.auc_roc),
        ("Precision (human)", human.get("precision")),
        ("Recall (human)", human.get("recall")),
        ("F1 (human)", human.get("f1")),
        ("Precision (ai)", ai.get("precision")),
        ("Recall (ai)", ai.get("recall")),
        ("F1 (ai)", ai.get("f1")),
    ]
    lines = ["| Metric         | Value |", "|----------------|-------|"]
    for label, value in rows:
        formatted = "TBD" if value is None else f"{value:.4f}"
        lines.append(f"| {label:<14} | {formatted} |")
    return "\n".join(lines)


def _splice_into_report(report_path: Path, table_md: str) -> None:
    """Replace the placeholder metrics table in the baseline report."""
    if not report_path.exists():
        logger.warning("Report not found at %s; skipping splice.", report_path)
        return
    text = report_path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"\| Metric\s+\| Value \|\n\|[-\s|]+\|\n(?:\|[^\n]*\|\n)+",
        re.MULTILINE,
    )
    if not pattern.search(text):
        logger.warning("No placeholder table found in %s; skipping splice.", report_path)
        return
    new_text = pattern.sub(table_md + "\n", text, count=1)
    report_path.write_text(new_text, encoding="utf-8")
    logger.info("Spliced baseline test metrics into %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print metrics JSON files and optionally update the baseline report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=config.REPORTS_DIR / "results_baseline.md",
    )
    parser.add_argument(
        "--baseline-test-json",
        type=Path,
        default=config.METRICS_DIR / "baseline_test.json",
        help="Path to the baseline test metrics JSON. Used for the splice step.",
    )
    parser.add_argument(
        "--update-baseline-md",
        action="store_true",
        help="Splice the baseline test metrics table into the baseline report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    if not args.metrics_dir.exists():
        logger.warning("Metrics dir %s does not exist; nothing to print.", args.metrics_dir)
        return 0

    metric_files = sorted(args.metrics_dir.glob("*.json"))
    if not metric_files:
        logger.info("No metric JSON files found under %s", args.metrics_dir)
    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "accuracy" in payload and "per_class" in payload:
            _print_metrics(path.name, payload)
        elif "n_errors" in payload:
            _print_error_summary(path.name, payload)
        else:
            print(f"\n=== {path.name} (raw) ===")
            print(json.dumps(payload, indent=2))

    csv_files = sorted(args.metrics_dir.glob("*.csv"))
    if csv_files:
        print("\n=== CSV artifacts ===")
        for path in csv_files:
            print(f"  {path.relative_to(_PROJECT_ROOT).as_posix()}")

    if args.update_baseline_md:
        if not args.baseline_test_json.exists():
            logger.warning(
                "Baseline test metrics JSON not found at %s; cannot splice into report.",
                args.baseline_test_json,
            )
        else:
            payload = json.loads(args.baseline_test_json.read_text(encoding="utf-8"))
            table = _build_baseline_test_table(payload)
            _splice_into_report(args.baseline_report, table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
