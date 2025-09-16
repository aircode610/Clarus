from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests, os, io

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("ASR_MODEL", "gpt-4o-mini-transcribe")  # or gpt-4o-transcribe, whisper-1

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
async def root():
    return FileResponse("whisper.html")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...),
                     response_format: str = Form("json"),
                     language: str | None = Form(None),
                     temperature: float = Form(0)):
    data = {
        "model": MODEL,
        "response_format": response_format,   # "json" | "verbose_json" | "srt" | "vtt"
        "temperature": str(temperature)
    }
    print(OPENAI_API_KEY)
    if language: data["language"] = language

    files = {
        "file": (file.filename, await file.read(), file.content_type or "audio/*")
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=headers, data=data, files=files, timeout=120
    )
    r.raise_for_status()
    # If response_format is srt/vtt return text; otherwise JSON.
    ct = r.headers.get("content-type", "")
    return r.text if "text/" in ct else r.json()