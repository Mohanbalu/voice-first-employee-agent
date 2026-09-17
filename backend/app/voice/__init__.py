"""Voice Processing Package — Speech-to-Text & Text-to-Speech (Module 6)."""

from backend.app.voice.speech_to_text import (
    DEFAULT_STT_MODEL,
    SUPPORTED_AUDIO_EXTENSIONS,
    GroqSpeechToTextProvider,
    MockSpeechToTextProvider,
    SpeechToTextProvider,
    STTError,
    STTTranscriptionResult,
    get_stt_provider,
)
from backend.app.voice.text_to_speech import (
    DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    SUPPORTED_ORPHEUS_VOICES,
    SUPPORTED_TTS_FORMATS,
    GroqTextToSpeechProvider,
    MockTextToSpeechProvider,
    TextToSpeechProvider,
    TTSError,
    TTSSynthesisResult,
    get_tts_provider,
)
from backend.app.voice.tts_service import TTSService
from backend.app.voice.voice_agent_service import VoiceAgentService

__all__ = [
    "DEFAULT_STT_MODEL",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "GroqSpeechToTextProvider",
    "MockSpeechToTextProvider",
    "SpeechToTextProvider",
    "STTError",
    "STTTranscriptionResult",
    "VoiceAgentService",
    "get_stt_provider",
    "DEFAULT_TTS_FORMAT",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TTS_VOICE",
    "SUPPORTED_ORPHEUS_VOICES",
    "SUPPORTED_TTS_FORMATS",
    "GroqTextToSpeechProvider",
    "MockTextToSpeechProvider",
    "TextToSpeechProvider",
    "TTSError",
    "TTSSynthesisResult",
    "get_tts_provider",
    "TTSService",
]
