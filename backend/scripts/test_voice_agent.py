"""Manual Integration Smoke Test for Module 6.2 (STT → LangGraph Agent Orchestrator).

Executes the full voice-first agent pipeline:
  Audio File → Groq Whisper STT → Transcript → AgentOrchestrator → RAG/AWS RDS pgvector → Groq LLM → Answer

Usage:
  py backend/scripts/test_voice_agent.py [path_to_audio_file]

Default audio file:
  voice.mp3 (in repository root)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
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
from backend.app.voice.voice_agent_service import VoiceAgentService
from backend.app.voice.speech_to_text import GroqSpeechToTextProvider, STTError
from backend.app.agents.orchestrator import AgentOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual smoke test for Module 6.2: Audio → STT → LangGraph Agent → RAG."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="voice.mp3",
        help="Path to audio recording (default: voice.mp3 in project root).",
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=config.tenant.default_id,
        help="Tenant UUID context (default: dev-org).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO-639-1 language code (e.g. 'en').",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        # Check relative to project root
        root = Path(__file__).resolve().parents[2]
        alt_path = root / args.audio_file
        if alt_path.exists():
            audio_path = alt_path
        else:
            print(f"\n[FAIL] Audio file not found: '{args.audio_file}'")
            print("Please specify an existing audio recording (e.g. voice.mp3).")
            sys.exit(1)

    print("=" * 60)
    print("        VOICE AGENT INTEGRATION TEST (MODULE 6.2)       ")
    print("=" * 60)
    print(f"Audio File:      {audio_path.name}")
    print(f"Audio Path:      {audio_path.resolve()}")
    print(f"File Size:       {audio_path.stat().st_size / 1024:.1f} KB")
    print(f"Tenant ID:       {args.tenant_id}")
    print(f"STT Model:       {config.stt.model} (Groq)")
    print(f"AI Provider:     {config.ai.provider} ({config.ai.groq_model})")
    print("-" * 60)
    print("Executing full pipeline: Audio → STT → LangGraph → RAG → LLM...")

    # Instantiate live services
    stt_provider = GroqSpeechToTextProvider(
        api_key=config.ai.groq_api_key,
        model=config.stt.model,
    )
    orchestrator = AgentOrchestrator()
    service = VoiceAgentService(stt_provider=stt_provider, orchestrator=orchestrator)

    total_start = time.perf_counter()
    try:
        result = service.process_voice_request(
            audio_file=audio_path,
            filename=audio_path.name,
            tenant_id=args.tenant_id,
            language=args.language,
        )
        elapsed_total = time.perf_counter() - total_start

        print("\n" + "=" * 60)
        print("VOICE AGENT INTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"Audio:          {audio_path.name}")
        print(f"STT Transcript: {result.transcript}")
        print(f"Intent:         {result.intent} (Confidence: {result.intent_confidence:.2f})")
        print(f"Agent Mode:     {result.agent_mode}")
        print(f"Status:         {result.status}")
        print(f"Elapsed:        {elapsed_total:.2f} seconds")
        print("-" * 60)
        print("Agent Response:")
        print(result.response)
        print("-" * 60)

        if result.rag_sources:
            print(f"Cited Sources ({len(result.rag_sources)}):")
            for idx, src in enumerate(result.rag_sources[:3], 1):
                doc_name = src.get("document_name", "Policy")
                page = src.get("page_start", "?")
                score = src.get("similarity_score", 0.0)
                print(f"  [{idx}] {doc_name} (Page {page}) — similarity: {score:.4f}")

        print("=" * 60)

    except STTError as exc:
        elapsed = time.perf_counter() - total_start
        print(f"\n[FAIL] STT error after {elapsed:.2f}s: {exc}")
        sys.exit(1)
    except Exception as exc:
        elapsed = time.perf_counter() - total_start
        print(f"\n[FAIL] Voice Agent pipeline error after {elapsed:.2f}s: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
