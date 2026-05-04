"""Classify a single code snippet as human-written or AI-generated.

Run as a script::

    uv run python src/ai_code_detector/cli/classify.py path/to/your_code.txt

or, equivalently, as a module::

    uv run python -m ai_code_detector.cli.classify path/to/your_code.txt

By default the script loads the checkpoint at ``models/baseline/``. Override
with ``--checkpoint-dir``. The decision threshold defaults to whatever was
recorded in ``training_config.json`` for that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# When invoked as a plain script (not via ``python -m``), make sure ``src/``
# is on the path so the package import below resolves regardless of the
# user's current working directory. This file lives at
# ``src/ai_code_detector/cli/classify.py``, so ``parents[2]`` is ``src/``.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.training.checkpoint import load_checkpoint  # noqa: E402
from ai_code_detector.training.loop import select_device  # noqa: E402

logger = logging.getLogger("classify")

_BAR_WIDTH: int = 20


def classify_code(
    code: str,
    *,
    checkpoint_dir: Path = config.BASELINE_MODEL_DIR,
    threshold: float | None = None,
    device_str: str | None = None,
) -> dict:
    """Run the classifier on a single code string and return the verdict.

    Args:
        code: The Python source code to classify.
        checkpoint_dir: Directory of the trained checkpoint. Defaults to
            ``models/baseline``.
        threshold: Decision threshold on ``P(ai)``. ``None`` means use the
            value stored in the checkpoint's ``training_config.json``.
        device_str: ``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``None`` (auto).

    Returns:
        Dict with keys ``label_name``, ``prediction``, ``prob_ai``,
        ``prob_human``, ``threshold``, ``n_chars``, and ``n_tokens``.
    """
    model, tokenizer, training_config = load_checkpoint(checkpoint_dir)
    decision_threshold = threshold if threshold is not None else training_config.decision_threshold

    device = select_device(device_str)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        code,
        truncation=True,
        max_length=training_config.max_sequence_length,
        padding=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    encoded = {key: tensor.to(device) for key, tensor in encoded.items()}

    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
    prob_ai = float(torch.sigmoid(output.logits).item())
    prediction = int(prob_ai >= decision_threshold)

    return {
        "label_name": config.LABEL_NAMES[prediction],
        "prediction": prediction,
        "prob_ai": prob_ai,
        "prob_human": 1.0 - prob_ai,
        "threshold": decision_threshold,
        "n_chars": len(code),
        "n_tokens": int(encoded["input_ids"].shape[1]),
    }


def _bar(probability: float, width: int = _BAR_WIDTH) -> str:
    """ASCII progress-bar of length ``width`` for a probability in [0, 1]."""
    filled = round(probability * width)
    return "[" + "#" * filled + " " * (width - filled) + "]"


def _print_human_readable(file_path: Path, result: dict) -> None:
    confidence = max(result["prob_ai"], result["prob_human"]) * 100.0
    label = result["label_name"].upper()
    print()
    print(f"  File      : {file_path}")
    print(f"  Length    : {result['n_chars']} chars, {result['n_tokens']} tokens")
    print(f"  Threshold : {result['threshold']:.3f}")
    print()
    print(f"  Verdict   : {label}-written  ({confidence:.1f}% confidence)")
    print()
    print(f"    P(ai)    {_bar(result['prob_ai'])} {result['prob_ai']:.4f}")
    print(f"    P(human) {_bar(result['prob_human'])} {result['prob_human']:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a single Python code snippet as human-written or AI-generated "
            "using the trained CodeBERT baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to a text file containing the code snippet to classify.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=config.BASELINE_MODEL_DIR,
        help="Directory containing the trained checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold on P(ai). Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a torch device (cpu, cuda, mps). Default: auto-detect.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as a single JSON line instead of human-readable output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(level=logging.WARNING)  # quiet by default; the verdict is the point

    if not args.file.exists():
        logger.error("File not found: %s", args.file)
        return 1
    if not args.file.is_file():
        logger.error("Not a regular file: %s", args.file)
        return 1

    code = args.file.read_text(encoding="utf-8")
    if not code.strip():
        logger.error("File is empty or whitespace-only: %s", args.file)
        return 2

    result = classify_code(
        code,
        checkpoint_dir=args.checkpoint_dir,
        threshold=args.threshold,
        device_str=args.device,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human_readable(args.file, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
