import os
import subprocess
import tempfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from voiceguard_inference import VoiceGuardAASIST


app = FastAPI(
    title="VoiceGuard AI API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voiceguard-lemon.vercel.app"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = VoiceGuardAASIST()


class VoiceGuardPredictionResponse(BaseModel):
    model: str
    sample_rate: int
    input_samples: int
    duration_seconds: float
    bona_fide_score: float
    spoof_score: float
    risk_score: float = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    security_action: Literal[
        "MONITOR",
        "WARN",
        "VERIFY",
        "PREVENT + ESCALATE",
    ]
    inference_seconds: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "voiceguard-ai",
        "model": "AASIST-L",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=VoiceGuardPredictionResponse)
async def predict(file: UploadFile = File(...)):
    allowed_extensions = {
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".ogg",
        ".webm",
    }

    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {extension}",
        )

    input_bytes = await file.read()

    if not input_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty.",
        )

    input_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as input_file:
            input_file.write(input_bytes)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as wav_file:
            wav_path = wav_file.name

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        import wave

        with wave.open(wav_path, "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        waveform = np.frombuffer(
            frames,
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

        result = MODEL.predict(waveform)

        result["audio_rms"] = float(
            np.sqrt(np.mean(waveform ** 2))
        )
        result["audio_peak"] = float(
            np.max(np.abs(waveform))
        )
        result["audio_mean"] = float(
            np.mean(waveform)
        )

        result["duration_seconds"] = (
            len(waveform) / sample_rate
        )

        return result

    except subprocess.CalledProcessError:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded audio file.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}",
        )

    finally:
        for path in (input_path, wav_path):
            if path and os.path.exists(path):
                os.remove(path)
