#!/usr/bin/env python3
"""Evaluate an openWakeWord ONNX model on labeled 16 kHz mono PCM16 WAVs."""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
from openwakeword.model import Model

FRAME_SAMPLES = 1280
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "reechy-spk150-steps100k-acc98.15-rec97.50.onnx"
FEATURE_DIR = ROOT / "models" / "openwakeword"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"{path}: must be 16 kHz / mono / PCM16 WAV")
        return np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")


def evaluate(model: Model, path: Path, threshold: float) -> tuple[float, list[tuple[float, float]]]:
    audio = read_wav(path)
    remainder = len(audio) % FRAME_SAMPLES
    if remainder:
        audio = np.pad(audio, (0, FRAME_SAMPLES - remainder))
    model.reset()
    peak = 0.0
    detections: list[tuple[float, float]] = []
    for offset in range(0, len(audio), FRAME_SAMPLES):
        score = max(float(value) for value in model.predict(audio[offset : offset + FRAME_SAMPLES]).values())
        peak = max(peak, score)
        if score >= threshold:
            detections.append(((offset + FRAME_SAMPLES) / 16000.0, score))
    return peak, detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, nargs="*", help="legacy positive samples")
    parser.add_argument("--positive", type=Path, action="append", default=[])
    parser.add_argument("--negative", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    positives = [*args.wav, *args.positive]
    negatives = args.negative
    if not positives and not negatives:
        parser.error("provide --positive/--negative WAVs (or positional positive WAVs)")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")

    model = Model(
        wakeword_models=[str(args.model)],
        inference_framework="onnx",
        melspec_model_path=str(FEATURE_DIR / "melspectrogram.onnx"),
        embedding_model_path=str(FEATURE_DIR / "embedding_model.onnx"),
    )
    tp = fn = fp = tn = 0
    for expected, paths in ((True, positives), (False, negatives)):
        for path in paths:
            peak, detections = evaluate(model, path, args.threshold)
            hit = bool(detections)
            tp += int(expected and hit); fn += int(expected and not hit)
            fp += int(not expected and hit); tn += int(not expected and not hit)
            label = "positive" if expected else "negative"
            verdict = "HIT" if hit else "MISS"
            times = ", ".join(f"{t:.2f}s/{score:.3f}" for t, score in detections[:5]) or "-"
            print(f"{label:8} {verdict:4} peak={peak:.3f} detections={times} {path}")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    print(f"TP={tp} FN={fn} FP={fp} TN={tn} recall={recall:.4f} FPR={fpr:.4f}")


if __name__ == "__main__":
    main()
