from pathlib import Path
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
from scipy.signal import butter, sosfilt
import onnxruntime as ort


class VoiceGuardAASIST:

    SAMPLE_RATE = 16000
    TARGET_SAMPLES = 64600

    def __init__(self, model_path=None):

        base_dir = Path(__file__).resolve().parent

        if model_path is None:
            model_path = (
                base_dir
                / "models"
                / "weights"
                / "AASIST-L.onnx"
            )

        self.model_path = str(model_path)

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        def reduce_background_noise(self, audio):
        audio = np.asarray(
            audio,
            dtype=np.float32
        ).reshape(-1)

        if len(audio) == 0:
            return audio

        # Lightweight high-pass filter to reduce
        # low-frequency background noise.
        sos = butter(
            4,
            80,
            btype="highpass",
            fs=self.SAMPLE_RATE,
            output="sos",
        )

        filtered = sosfilt(
            sos,
            audio
        )

        return filtered.astype(np.float32)
    def prepare_audio(self, audio):

        audio = np.asarray(
            audio,
            dtype=np.float32
        ).reshape(-1)

        if len(audio) == 0:
            raise ValueError("Audio input is empty.")

        

        if len(audio) >= self.TARGET_SAMPLES:

            processed = audio[:self.TARGET_SAMPLES]

        else:

            repeats = int(
                np.ceil(
                    self.TARGET_SAMPLES / len(audio)
                )
            )

            processed = np.tile(
                audio,
                repeats
            )[:self.TARGET_SAMPLES]

        return processed.astype(np.float32)

    def predict(self, audio):

        processed = self.prepare_audio(audio)

        start = time.perf_counter()

        logits = self.session.run(
            [self.output_name],
            {
                self.input_name:
                    processed[None, :].astype(np.float32)
            },
        )[0][0]

        inference_seconds = time.perf_counter() - start

        if logits.ndim != 1 or logits.shape[0] != 2:
            raise RuntimeError(
                f"Unexpected AASIST logits shape: {logits.shape}"
            )

        # AASIST-L convention:
        # logits[1] = bona-fide score.
        bona_fide_score = float(logits[1])

        # VoiceGuard convention:
        # higher spoof_score = more suspicious.
        spoof_score = -bona_fide_score

        risk = score_to_risk(spoof_score)
        level = risk_level(risk)
        action = security_action(level)

        return {
            "model": "AASIST-L",
            "sample_rate": self.SAMPLE_RATE,
            "input_samples": self.TARGET_SAMPLES,
            "bona_fide_score": round(
                bona_fide_score,
                6
            ),
            "spoof_score": round(
                spoof_score,
                6
            ),
            "risk_score": round(
                risk,
                2
            ),
            "risk_level": level,
            "security_action": action,
            "inference_seconds": round(
                inference_seconds,
                4
            ),
        }


# VoiceGuard calibrated risk policy
#
# Calibration source:
# 200-sample balanced ASVspoof evaluation
#
# Operating points:
#   0.52  ->   0 risk
#   2.25  ->  50 risk
#   3.75  ->  80 risk
#   4.94  -> 100 risk
#
# These values are an initial benchmark-calibrated policy,
# not universal scientific risk boundaries.

RISK_LOW_ANCHOR = 0.52
RISK_THRESHOLD = 2.25
RISK_HIGH_ANCHOR = 3.75
RISK_CRITICAL_ANCHOR = 4.94


def score_to_risk(spoof_score):

    x = float(spoof_score)

    if x <= RISK_LOW_ANCHOR:
        return 0.0

    elif x <= RISK_THRESHOLD:
        return (
            (x - RISK_LOW_ANCHOR)
            / (RISK_THRESHOLD - RISK_LOW_ANCHOR)
        ) * 50.0

    elif x <= RISK_HIGH_ANCHOR:
        return 50.0 + (
            (x - RISK_THRESHOLD)
            / (RISK_HIGH_ANCHOR - RISK_THRESHOLD)
        ) * 30.0

    else:
        return min(
            100.0,
            80.0 + (
                (x - RISK_HIGH_ANCHOR)
                / (RISK_CRITICAL_ANCHOR - RISK_HIGH_ANCHOR)
            ) * 20.0
        )


def risk_level(risk_score):

    if risk_score <= 30:
        return "LOW"

    elif risk_score <= 60:
        return "MEDIUM"

    elif risk_score <= 80:
        return "HIGH"

    else:
        return "CRITICAL"


def security_action(level):

    if level == "LOW":
        return "MONITOR"

    elif level == "MEDIUM":
        return "WARN"

    elif level == "HIGH":
        return "VERIFY"

    else:
        return "PREVENT + ESCALATE"
