import os
import io
import zipfile
import requests
import streamlit as st
from audiorecorder import audiorecorder
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

from faster_whisper import WhisperModel
import json

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

def get_whisper_model(lang_key: str):
    model = WhisperModel("base", device="cpu")
    segments, info = model.transcribe("sample.wav")
    return segments

def voice_to_text():
    audio = audiorecorder(
        start_prompt="Start recording",
        stop_prompt="Stop",
        pause_prompt=None,
    )

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


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size="small", device="auto", compute_type="int8"):
    """
    model_size: tiny | base | small | medium | large-v2 | large-v3 | distil-... (any HF repo supported)
    device: "cpu", "cuda", "auto"
    compute_type: "int8", "int8_float16", "float16", "float32" (pick what's supported by your device)
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def whisper_voice_to_text(
    lang_key=None,          # e.g., "en", "de", "ru" — faster-whisper will auto-detect if None
    append_mode=True,
    insert_newline=True,
    auto_punct=True,        # unused; Whisper already handles punctuation
    model_size="small",
    device="auto",
    compute_type="int8",
    beam_size=5,
    vad_filter=True,
):
    audio = audiorecorder(
        start_prompt="Start recording",
        stop_prompt="Stop",
        pause_prompt=None,
    )

    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    if len(audio) > 0:
        # prepare 16k mono PCM wav bytes (fine for Whisper)
        seg: AudioSegment = (
            audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        )
        wav_buf = io.BytesIO()
        seg.export(wav_buf, format="wav")
        wav_bytes = wav_buf.getvalue()

        # load/cached faster-whisper model
        model = load_whisper_model(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
        )

        # Transcribe. You can also pass a file-like object directly.
        # language=lang_key pins language; leave None for auto-detect
        segments, info = model.transcribe(
            io.BytesIO(wav_bytes),
            language=lang_key,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=False,  # set True if you need per-word timings
        )

        # Collect text. Segment.text usually includes leading spaces—concat then strip.
        new_transcript = "".join(s.text for s in segments).strip()

        if append_mode:
            prev = st.session_state.get("transcript", "")
            sep = "\n" if insert_newline and prev and new_transcript else (" " if prev and new_transcript else "")
            st.session_state.transcript = f"{prev}{sep}{new_transcript}"
        else:
            st.session_state.transcript = new_transcript


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

lang_key = "English (en-US)"
auto_punct = True
append_mode = True
insert_newline = True

def main():
    global lang_key, auto_punct, append_mode, insert_newline

    st.set_page_config(page_title="Speech → Text (Free)", page_icon="🎙️", layout="centered")
    st.title("🎙️ Speech → Text (Free, offline)")
    st.caption("Press **Start**, speak, then **Stop**. Your speech is transcribed to text below. No paid APIs.")

    # --- Sidebar options ---
    st.sidebar.header("Settings")
    lang_key = st.sidebar.selectbox("Recognition language", list(VOSK_MODELS.keys()), index=0)
    auto_punct = st.sidebar.checkbox("Enable auto punctuation (if model supports)", value=True)
    append_mode = st.sidebar.checkbox("Append new take to existing text", value=True)
    insert_newline = st.sidebar.checkbox("Insert a newline between takes", value=True)

    # --- Recorder UI ---
    st.subheader("Recorder")
    st.write("Click **Start recording**, speak, then click **Stop**.")

    whisper_voice_to_text()

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

if __name__ == "__main__":
    main()
