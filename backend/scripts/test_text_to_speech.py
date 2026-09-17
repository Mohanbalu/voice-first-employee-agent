"""Standalone CLI Smoke Test for Text-to-Speech Subsystem (Module 6.3).

Tests speech synthesis with Groq Orpheus TTS or offline Mock TTS,
measures elapsed latency, and saves the output audio to data/processed/tts/.

Usage:
  py backend/scripts/test_text_to_speech.py "Hello, welcome to the employee assistant."
  py backend/scripts/test_text_to_speech.py --mock "Testing offline mock speech synthesis."
  py backend/scripts/test_text_to_speech.py --voice diana "Welcome to our team."
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import config
from backend.app.voice.text_to_speech import (
    DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    GroqTextToSpeechProvider,
    MockTextToSpeechProvider,
    TTSError,
    TTSSynthesisResult,
    get_tts_provider,
)
from backend.app.voice.tts_service import TTSService
from backend.app.schemas.voice import SynthesizeRequest


def mask_secret(secret: str | None) -> str:
    """Safely masks secret tokens for reporting."""
    if not secret:
        return "(None configured)"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:6]}...{secret[-4:]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Groq Text-to-Speech Live Smoke Test")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello, welcome to the employee assistant.",
        help="Text string to synthesize into speech.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use offline Mock TTS provider instead of live Groq API.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice persona to use (e.g. autumn, diana, hannah, austin, troy).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="TTS model identifier (default: canopylabs/orpheus-v1-english).",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Audio format (default: wav).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "tts"),
        help="Directory to save synthesized audio files.",
    )

    args = parser.parse_args()

    # Determine provider and settings
    if args.mock or (os.getenv("TTS_PROVIDER", "").strip().lower() == "mock"):
        provider_name = "Mock (Offline)"
        provider = MockTextToSpeechProvider()
    else:
        provider_name = "Groq Text-to-Speech"
        provider = GroqTextToSpeechProvider(
            model=args.model,
            voice=args.voice,
            format=args.format,
        )

    model_name = args.model or getattr(provider, "model_name", DEFAULT_TTS_MODEL)
    voice_name = args.voice or getattr(provider, "default_voice", DEFAULT_TTS_VOICE)
    format_name = args.format or DEFAULT_TTS_FORMAT

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("           GROQ TEXT-TO-SPEECH SMOKE TEST            ")
    print("=" * 60)
    print(f"Provider:        {provider_name}")
    print(f"Model:           {model_name}")
    print(f"Voice:           {voice_name}")
    print(f"Format:          {format_name}")
    if not args.mock:
        print(f"API Key:         {mask_secret(config.ai.groq_api_key)}")
    print(f"Text:            {args.text}")
    print(f"Text Length:     {len(args.text)} characters")
    print("-" * 60)
    print("Generating speech...")

    start_time = time.time()
    try:
        service = TTSService(provider=provider)
        request = SynthesizeRequest(
            text=args.text,
            voice=args.voice,
            model=args.model,
            format=args.format,
        )
        result: TTSSynthesisResult = service.synthesize_speech(request)
        elapsed = time.time() - start_time

        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}.{result.format}"
        output_file = output_dir / filename
        output_file.write_bytes(result.audio_bytes)

        file_size = len(result.audio_bytes)

        print("\n[SUCCESS] Speech synthesis completed!")
        print(f"Output:          {output_file.relative_to(PROJECT_ROOT) if output_file.is_relative_to(PROJECT_ROOT) else output_file}")
        print(f"File Size:       {file_size:,} bytes ({file_size / 1024:.1f} KB)")
        print(f"Elapsed Time:    {elapsed:.2f} seconds")
        print("=" * 60)
        return 0

    except TTSError as exc:
        elapsed = time.time() - start_time
        print(f"\n[BLOCKED / FAILED] TTS synthesis could not complete (Elapsed: {elapsed:.2f}s)")
        print(f"Status Code:     {exc.status_code}")
        print(f"Reason:          {exc}")

        if "terms acceptance" in str(exc).lower():
            print("\n" + "=" * 60)
            print("ACTION REQUIRED:")
            print("Groq's official TTS model 'canopylabs/orpheus-v1-english' requires")
            print("one-time terms of service acceptance in the Groq console.")
            print("Please visit the URL below to accept the terms:")
            print("  https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english")
            print("After acceptance, rerun this test to generate live speech.")
            print("You can also run offline testing anytime with: --mock")
            print("=" * 60)
        return 1

    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Unexpected failure during TTS smoke test (Elapsed: {elapsed:.2f}s)")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
