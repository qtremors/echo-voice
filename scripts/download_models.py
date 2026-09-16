"""Download the default lightweight Whisper and Piper models."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", default="tiny.en")
    parser.add_argument("--piper", default="en_US-lessac-low")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Whisper {args.whisper}...", flush=True)
    from huggingface_hub import snapshot_download

    whisper_repo = (
        "Systran/faster-whisper-tiny.en" if args.whisper == "tiny.en" else args.whisper
    )
    snapshot_download(
        whisper_repo,
        local_dir=args.models_dir / "whisper-tiny-en",
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.bin",
            "tokenizer.json",
            "vocabulary.*",
        ],
    )

    print(f"Downloading Piper {args.piper}...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", args.piper],
        cwd=args.models_dir,
        check=True,
    )
    print("Models are ready.", flush=True)


if __name__ == "__main__":
    main()
