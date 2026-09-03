from pathlib import Path

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import ast
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from models.AASIST import Model 


class VoiceGuardAASIST:

    SAMPLE_RATE = 16000
    TARGET_SAMPLES = 64600

    def __init__(
        self,
        config_path="config/AASIST-L.conf",
        checkpoint_path="models/weights/AASIST-L.pth",
        device=None,
    ):

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # AASIST .conf files contain a Python-style dictionary.
        base_dir = Path(__file__).resolve().parent
        if config_path is None:
            config_path = str(base_dir / "config" / "AASIST-L.conf")
        if checkpoint_path is None:
            checkpoint_path = str(base_dir / "models" / "weights" / "AASIST-L.pth")

        with open(config_path, "r") as f:
            config = ast.literal_eval(f.read())
            model_config = config["model_config"]

        self.model = Model(model_config)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]

        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

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

        waveform = torch.tensor(
            processed,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        start = time.time()

        with torch.no_grad():
            hidden, logits = self.model(waveform)

        inference_seconds = time.time() - start

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected AASIST logits shape: {logits.shape}"
            )

        # AASIST-L convention:
        # logits[:, 1] = bona-fide score.
        bona_fide_score = float(
            logits[0, 1].item()
        )

        # VoiceGuard convention:
        # higher = more suspicious.
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
    """
    Convert AASIST-L spoof evidence into a 0-100
    VoiceGuard impersonation-risk score.

    Higher spoof_score = higher suspiciousness.
    """

    x = float(spoof_score)

    # Below the calibrated low-risk anchor.
    if x <= RISK_LOW_ANCHOR:
        return 0.0

    # 0.52 -> 2.25 maps to 0 -> 50.
    elif x <= RISK_THRESHOLD:
        return (
            (x - RISK_LOW_ANCHOR)
            / (RISK_THRESHOLD - RISK_LOW_ANCHOR)
        ) * 50.0

    # 2.25 -> 3.75 maps to 50 -> 80.
    elif x <= RISK_HIGH_ANCHOR:
        return 50.0 + (
            (x - RISK_THRESHOLD)
            / (RISK_HIGH_ANCHOR - RISK_THRESHOLD)
        ) * 30.0

    # 3.75 -> 4.94 maps to 80 -> 100.
    else:
        return min(
            100.0,
            80.0 + (
                (x - RISK_HIGH_ANCHOR)
                / (RISK_CRITICAL_ANCHOR - RISK_HIGH_ANCHOR)
            ) * 20.0
        )


def risk_level(risk_score):
    """Convert numerical risk into VoiceGuard risk level."""

    if risk_score <= 30:
        return "LOW"

    elif risk_score <= 60:
        return "MEDIUM"

    elif risk_score <= 80:
        return "HIGH"

    else:
        return "CRITICAL"


def security_action(level):
    """Map VoiceGuard risk level to the corresponding response."""

    if level == "LOW":
        return "MONITOR"

    elif level == "MEDIUM":
        return "WARN"

    elif level == "HIGH":
        return "VERIFY"

    else:
        return "PREVENT + ESCALATE"
