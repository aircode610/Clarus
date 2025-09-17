import os
import io
import zipfile
import requests
import streamlit as st
from audiorecorder import audiorecorder
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json

st.set_page_config(page_title="Speech → Text (Free)", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech → Text (Free, offline)")
st.caption("Press **Start**, speak, then **Stop**. Your speech is transcribed to text below. No paid APIs.")

# --- Model registry (small models) ---
VOSK_MODELS = {
    "English (en-US)": {
        "code": "en-us",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip",
        "dirname": "vosk-model-en-us-0.22-lgraph",
    },
    "German (de)": {
        "code": "de",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
        "dirname": "vosk-model-small-de-0.15",
    },
    "Spanish (es)": {
        "code": "es",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
        "dirname": "vosk-model-small-es-0.42",
    },
    "French (fr)": {
        "code": "fr",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
        "dirname": "vosk-model-small-fr-0.22",
    },
}

MODELS_DIR = os.path.join(".models", "vosk")
os.makedirs(MODELS_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def get_vosk_model(lang_key: str) -> Model:
    """Download (if needed) and load a Vosk model."""
    meta = VOSK_MODELS[lang_key]
    model_dir = os.path.join(MODELS_DIR, meta["dirname"])
    if not os.path.isdir(model_dir):
        with st.spinner(f"Downloading {lang_key} model (~50–90 MB)…"):
            zip_path = os.path.join(MODELS_DIR, meta["dirname"] + ".zip")
            with requests.get(meta["url"], stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(MODELS_DIR)
            os.remove(zip_path)
            if not os.path.isdir(model_dir):
                for name in os.listdir(MODELS_DIR):
                    cand = os.path.join(MODELS_DIR, name)
                    if os.path.isdir(cand) and meta["dirname"] in name:
                        model_dir = cand
                        break
    return Model(model_dir)

# --- Sidebar options ---
st.sidebar.header("Settings")
lang_key = st.sidebar.selectbox("Recognition language", list(VOSK_MODELS.keys()), index=0)
auto_punct = st.sidebar.checkbox("Enable auto punctuation (if model supports)", value=True)
append_mode = st.sidebar.checkbox("Append new take to existing text", value=True)
insert_newline = st.sidebar.checkbox("Insert a newline between takes", value=True)

# --- Recorder UI ---
st.subheader("Recorder")
st.write("Click **Start recording**, speak, then click **Stop**.")

audio = audiorecorder(
    start_prompt="Start recording",
    stop_prompt="Stop",
    pause_prompt=None,
)

# --- Transcription state ---
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if len(audio) > 0:
    seg: AudioSegment = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    wav_buf = io.BytesIO()
    seg.export(wav_buf, format="wav")
    wav_bytes = wav_buf.getvalue()

    model = get_vosk_model(lang_key)

    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    try:
        if auto_punct:
            pass
    except Exception:
        pass

    import wave
    wf = wave.open(io.BytesIO(wav_bytes), "rb")
    result_text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            j = json.loads(rec.Result())
            if j.get("text"):
                result_text.append(j["text"])
    j = json.loads(rec.FinalResult())
    if j.get("text"):
        result_text.append(j["text"])

    new_transcript = (" ".join(result_text)).strip()
    if append_mode:
        prev = st.session_state.get("transcript", "")
        sep = "\n" if insert_newline and prev and new_transcript else (" " if prev and new_transcript else "")
        st.session_state.transcript = f"{prev}{sep}{new_transcript}"
    else:
        st.session_state.transcript = new_transcript

st.subheader("Transcript")
st.text_area("text", label_visibility="collapsed", value=st.session_state.transcript, height=200)

st.info("Tip: If your mic access is blocked by the browser, allow microphone permissions and reload the page.")

st.markdown(
    """
---
**Notes**
- Models by Vosk / AlphaCephei (Apache 2.0). Everything runs locally; no cloud costs.
- For best accuracy, speak clearly near your mic.
- You can swap to larger Vosk models for higher accuracy—just add them to `VOSK_MODELS`.
    """
)