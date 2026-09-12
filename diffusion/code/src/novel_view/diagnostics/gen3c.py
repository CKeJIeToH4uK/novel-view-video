"""Явная двухпроцессная Gen3C model-ready диагностика."""

from __future__ import annotations

import argparse
from pathlib import Path

from novel_view.runtime.executables import GEN3C_PYTHON
from novel_view.runtime.process import run_process


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--magnitude-m", type=float, required=True)
    parser.add_argument("--sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--target-rgb", type=Path, required=True)
    parser.add_argument("--condition-rgb", type=Path, required=True)
    parser.add_argument("--condition-known", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cuda-visible-device", required=True)
    parser.add_argument("--vae-only", action="store_true")
    return parser.parse_args()


def _run(arguments: argparse.Namespace) -> None:
    environment = {"CUDA_VISIBLE_DEVICES": arguments.cuda_visible_device}
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("_gen3c_model_ready_worker.py")
    if not arguments.vae_only:
        run_process(
            [
                str(GEN3C_PYTHON),
                str(worker),
                "prompt",
                "--text-encoder",
                str(arguments.text_encoder),
                "--scratch-root",
                str(arguments.scratch_root),
            ],
            arguments.run_dir / "prompt.log",
            "Gen3C model-ready prompt canary",
            environment,
        )
    run_process(
        [
            str(GEN3C_PYTHON),
            str(worker),
            "vae",
            "--sample-id",
            arguments.sample_id,
            "--magnitude-m",
            str(arguments.magnitude_m),
            "--sign",
            str(arguments.sign),
            "--target-rgb",
            str(arguments.target_rgb),
            "--condition-rgb",
            str(arguments.condition_rgb),
            "--condition-known",
            str(arguments.condition_known),
            "--tokenizer",
            str(arguments.tokenizer),
            "--scratch-root",
            str(arguments.scratch_root),
        ],
        arguments.run_dir / "vae.log",
        "Gen3C model-ready VAE canary",
        environment,
    )


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()
