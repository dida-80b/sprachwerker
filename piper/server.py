#!/usr/bin/env python3
import io
import logging
import os
import wave
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sprachwerker-piper")

VOICES_DIR = Path(os.getenv("VOICES_DIR", "/voices"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "de_DE-thorsten-high")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5150"))


def get_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        for ep in ("ROCMExecutionProvider", "MIGraphXExecutionProvider"):
            if ep in available:
                logger.info("ONNX: using %s", ep)
                return [ep, "CPUExecutionProvider"]
    except Exception:
        pass
    logger.info("ONNX: using CPUExecutionProvider")
    return ["CPUExecutionProvider"]


ONNX_PROVIDERS = get_onnx_providers()
loaded_voices: dict[str, object] = {}


def load_all_voices() -> None:
    from piper import PiperVoice

    if not VOICES_DIR.exists():
        logger.warning("Voices dir not found: %s", VOICES_DIR)
        return

    for onnx_path in sorted(VOICES_DIR.glob("*.onnx")):
        name = onnx_path.stem
        json_path = onnx_path.with_suffix(".onnx.json")
        if not json_path.exists():
            logger.warning("Missing config for %s, skipping", onnx_path.name)
            continue
        try:
            use_cuda = any("ROCM" in p or "CUDA" in p for p in ONNX_PROVIDERS)
            voice = PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=use_cuda)
            loaded_voices[name] = voice
            logger.info("Loaded voice: %s", name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", name, exc)


app = FastAPI(title="Sprachwerker Piper", version="1.0.0")


class SynthRequest(BaseModel):
    text: str
    voice: Optional[str] = None


@app.on_event("startup")
async def startup() -> None:
    load_all_voices()
    logger.info("Piper ready with %d voice(s): %s", len(loaded_voices), list(loaded_voices))


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "voices": list(loaded_voices)}


@app.get("/voices")
def voices() -> dict[str, object]:
    return {"voices": list(loaded_voices)}


@app.post("/synthesize")
def synthesize(req: SynthRequest) -> Response:
    voice_name = req.voice or DEFAULT_VOICE or (list(loaded_voices)[0] if loaded_voices else None)
    if not voice_name:
        raise HTTPException(503, "No voices loaded")

    voice = loaded_voices.get(voice_name)
    if not voice:
        for key in loaded_voices:
            if voice_name in key or key in voice_name:
                voice = loaded_voices[key]
                voice_name = key
                break
    if not voice:
        raise HTTPException(404, f"Voice '{req.voice}' not found. Available: {list(loaded_voices)}")

    try:
        buffer = io.BytesIO()
        first_chunk = None
        pcm_parts: list[bytes] = []
        for chunk in voice.synthesize(req.text):
            if first_chunk is None:
                first_chunk = chunk
            pcm_parts.append(chunk.audio_int16_bytes)
        if not first_chunk or not pcm_parts:
            raise ValueError("No audio generated")
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(first_chunk.sample_channels)
            wav.setsampwidth(first_chunk.sample_width)
            wav.setframerate(first_chunk.sample_rate)
            for pcm in pcm_parts:
                wav.writeframes(pcm)
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        raise HTTPException(500, f"Synthesis error: {exc}")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
