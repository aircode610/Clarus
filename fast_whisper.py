import asyncio
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, List
import re

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from faster_whisper import WhisperModel
except Exception as e:  # pragma: no cover
    WhisperModel = None  # type: ignore

APP_DIR = Path(__file__).parent
HTML_FILE = APP_DIR / "fast_whisper.html"

app = FastAPI(title="Fast Whisper Live Translation")

# Allow local dev from different ports if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once at startup
_model: Optional[WhisperModel] = None
_model_lock = asyncio.Lock()


async def get_model() -> WhisperModel:
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                if WhisperModel is None:
                    raise RuntimeError("faster-whisper not installed. Add 'faster-whisper' to requirements and pip install.")
                model_size = os.getenv("FW_MODEL", "small")
                device = os.getenv("FW_DEVICE", "cpu")
                compute_type = os.getenv("FW_COMPUTE", "int8")  # int8 for CPU-friendly
                # You can set environment vars to tune model
                _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model  # type: ignore


@app.get("/")
async def index() -> HTMLResponse:
    if not HTML_FILE.exists():
        return HTMLResponse("<h1>fast_whisper.html not found</h1>", status_code=404)
    return HTMLResponse(HTML_FILE.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok")


def _read_wav_pcm16_mono(data: bytes) -> tuple[np.ndarray, int]:
    """
    Minimal WAV reader for PCM16 mono; returns float32 numpy array in [-1,1] and sample_rate.
    Accepts chunks that are standalone small WAV files.
    """
    import wave

    with wave.open(BytesIO(data), "rb") as wf:
        num_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM supported, got sampwidth={sampwidth}")
    if num_channels < 1:
        raise ValueError("No channels in WAV chunk")

    # If stereo or more, take channel 0 (interleaved)
    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    if num_channels > 1:
        audio_i16 = audio_i16.reshape(-1, num_channels)[:, 0]

    audio_f32 = (audio_i16.astype(np.float32)) / 32768.0
    return audio_f32, framerate


def _resample_to_16k(audio_f32: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample mono float32 audio to 16 kHz using linear interpolation (no extra deps)."""
    target_sr = 16000
    if sample_rate == target_sr:
        return audio_f32.astype(np.float32, copy=False)
    n = len(audio_f32)
    if n == 0:
        return audio_f32.astype(np.float32, copy=False)
    # Map original sample indices to new ones
    new_len = int(round(n * (target_sr / float(sample_rate))))
    if new_len <= 1:
        return audio_f32.astype(np.float32, copy=False)
    x_old = np.arange(n, dtype=np.float32)
    x_new = np.linspace(0, n - 1, new_len, dtype=np.float32)
    y_new = np.interp(x_new, x_old, audio_f32).astype(np.float32)
    return y_new


def _merge_append(full_text: str, addition: str, window_chars: int = 400, min_overlap: int = 10) -> str:
    """Append addition to full_text while preventing duplicates due to window overlap.
    - Normalizes whitespace for matching.
    - If the normalized addition is already inside the recent tail of full_text, skip.
    - Otherwise, find the longest suffix of full_text's tail that matches a prefix of addition and append only the novel suffix.
    Note: Appends a whitespace-normalized suffix to keep logic simple.
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    if not addition:
        return full_text

    if not full_text:
        return norm(addition)

    fn = norm(full_text)
    an = norm(addition)
    if not an:
        return full_text

    tail = fn[-window_chars:]
    # If the whole addition is already in the tail, skip
    if an in tail:
        return full_text

    # Find longest overlap between tail suffix and addition prefix
    max_search = min(len(an), len(tail))
    overlap = 0
    for k in range(max_search, min_overlap - 1, -1):
        if tail.endswith(an[:k]):
            overlap = k
            break

    suffix = an[overlap:] if overlap > 0 else an
    if not suffix:
        return full_text

    # Append with a space if needed
    joiner = " " if fn and not fn.endswith(" ") and not suffix.startswith(" ") else ""
    return (fn + joiner + suffix).strip()


async def _transcribe_all(audio_f32: np.ndarray, sample_rate: int, translate: bool = True) -> str:
    """Run full transcription on current buffer and return concatenated text."""
    model = await get_model()

    # Ensure 16 kHz for numpy-array input; faster-whisper expects 16k when not using ffmpeg paths
    audio_16k = _resample_to_16k(audio_f32, sample_rate)
    task = "translate" if translate else "transcribe"

    loop = asyncio.get_running_loop()

    def _do_transcribe() -> str:
        segments, info = model.transcribe(audio_16k, task=task)
        texts: List[str] = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text.strip())
        return " ".join(t for t in texts if t)

    return await loop.run_in_executor(None, _do_transcribe)


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()

    # End-only mode: buffer all audio during recording; transcribe once after [END]
    translate = True
    target_sr = 16000
    chunks: List[np.ndarray] = []  # float32 mono at 16 kHz

    try:
        while True:
            msg = await ws.receive()
            # Control messages
            if "text" in msg and msg["text"] is not None:
                text = (msg["text"] or "").strip()
                if text == "[END]":
                    # Concatenate and run a single transcription
                    audio_16k = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
                    try:
                        final_text = await _transcribe_all(audio_16k, target_sr, translate=translate)
                        await ws.send_text(f"[FINAL] {final_text}")
                    except Exception as e:
                        try:
                            await ws.send_text(f"[ERROR] {type(e).__name__}: {e}")
                        except Exception:
                            pass
                    break
                # ignore other text messages
                continue

            # Binary audio chunk (WAV PCM16 mono)
            data = msg.get("bytes")
            if not data:
                continue
            try:
                chunk_f32, sr = _read_wav_pcm16_mono(data)
                # Resample each chunk to 16 kHz to keep everything consistent
                if sr != target_sr:
                    chunk_f32 = _resample_to_16k(chunk_f32, sr)
                chunks.append(chunk_f32.astype(np.float32, copy=False))
            except Exception as e:
                try:
                    await ws.send_text(f"[ERROR] {type(e).__name__}: {e}")
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(f"[ERROR] {type(e).__name__}: {e}")
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Run with: python fast_whisper.py
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("fast_whisper:app", host="0.0.0.0", port=port, reload=False)
