"""Manual Smoke Test Script for Groq Speech-to-Text (Module 6.1).

Validates live audio transcription using Groq's official Whisper Large V3 Turbo API.

Usage:
  py backend/scripts/test_speech_to_text.py <path_to_audio_file>
  py backend/scripts/test_speech_to_text.py --generate-sample

Examples:
  py backend/scripts/test_speech_to_text.py data/sample_audio.wav
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root in sys.path
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from backend.app.config import config
from backend.app.voice.speech_to_text import (
    DEFAULT_STT_MODEL,
    SUPPORTED_AUDIO_EXTENSIONS,
    GroqSpeechToTextProvider,
    STTError,
)


def create_synthetic_wav(output_path: Path, duration_seconds: float = 2.0) -> Path:
    """Generates a simple 16-bit PCM mono 16kHz WAV file with a dual-tone chirp."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate a gentle audible dual-tone
        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # 440 Hz (A4) + 880 Hz (A5)
            val = int(10000 * (0.6 * math.sin(2 * math.pi * 440 * t) + 0.4 * math.sin(2 * math.pi * 880 * t)))
            frames.extend(struct.pack("<h", max(-32767, min(32767, val))))

        wav_file.writeframes(frames)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual smoke test for Groq Speech-to-Text (Whisper Large V3 Turbo)."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to an audio file (wav, mp3, m4a, webm, flac, etc.) to transcribe.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO-639-1 language code (e.g., 'en', 'es', 'fr').",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional context prompt to guide transcription.",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic test WAV file if you don't have one available.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("        GROQ SPEECH-TO-TEXT MANUAL SMOKE TEST        ")
    print("=" * 60)
    print(f"Provider:        Groq Speech-to-Text")
    print(f"Model:           {config.stt.model}")
    print(f"Base URL:        {config.ai.groq_base_url}")

    # 1. Verify API Key exists
    api_key = config.ai.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_placeholder":
        print("\n[FAIL] GROQ_API_KEY is not configured.")
        print("Please set GROQ_API_KEY in backend/.env to run the live smoke test.")
        sys.exit(1)

    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"API Key:         {masked_key} (Verified Present)")
    print("-" * 60)

    # 2. Check audio file argument
    target_audio: Optional[Path] = None
    if args.generate_sample or (args.audio_file and args.audio_file == "generate"):
        sample_path = Path("data/processed/sample_test.wav")
        create_synthetic_wav(sample_path)
        target_audio = sample_path
        print(f"[INFO] Generated synthetic test audio: {target_audio}")
    elif args.audio_file:
        target_audio = Path(args.audio_file)
        if not target_audio.exists():
            print(f"\n[FAIL] Audio file not found: {target_audio}")
            sys.exit(1)
    else:
        # Check if a default sample exists in data/
        possible_samples = list(Path("data").glob("**/*.wav")) + list(Path("data").glob("**/*.mp3"))
        if possible_samples:
            target_audio = possible_samples[0]
            print(f"[INFO] Found existing audio file: {target_audio}")
        else:
            print("\nUsage Instructions:")
            print("  py backend/scripts/test_speech_to_text.py <path_to_audio_file>")
            print("  py backend/scripts/test_speech_to_text.py --generate-sample")
            print(f"\nSupported audio formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}")
            print("\nNo audio file was specified. Pass a recording to test transcription.")
            sys.exit(0)

    # Validate format
    ext = target_audio.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        print(f"\n[FAIL] Unsupported audio extension: '.{ext}'")
        print(f"Supported formats: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}")
        sys.exit(1)

    file_size_bytes = target_audio.stat().st_size
    file_size_kb = file_size_bytes / 1024
    print(f"Audio File:      {target_audio.name}")
    print(f"Audio Path:      {target_audio.resolve()}")
    print(f"File Size:       {file_size_kb:.1f} KB ({file_size_bytes} bytes)")
    print("-" * 60)
    print("Dispatching transcription request to Groq Whisper...")

    provider = GroqSpeechToTextProvider(api_key=api_key, model=config.stt.model)

    start_time = time.perf_counter()
    try:
        result = provider.transcribe(
            audio_file=target_audio,
            language=args.language,
            prompt=args.prompt,
        )
        elapsed_seconds = time.perf_counter() - start_time

        print("\n[SUCCESS] Transcription completed!")
        print(f"Elapsed Time:    {elapsed_seconds:.2f}s")
        print(f"Language:        {result.language or 'auto'}")
        if result.duration:
            print(f"Duration:        {result.duration:.2f}s")
        print("\n--- Transcript ---")
        print(result.text if result.text else "(Empty transcript / no speech detected)")
        print("------------------")
        print("=" * 60)

    except STTError as exc:
        elapsed_seconds = time.perf_counter() - start_time
        print(f"\n[FAIL] Transcription error ({elapsed_seconds:.2f}s): {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
