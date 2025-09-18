"""
Voice processing package for Clarus application.

This package contains voice input functionality including:
- Speech-to-text transcription using Whisper
- Voice recording and processing
- Audio format conversion
"""

# Always import the voice module - it handles missing dependencies internally
from .streamlit_voice import whisper_voice_to_text, load_whisper_model

__all__ = [
    "whisper_voice_to_text",
    "load_whisper_model"
]
