"""Manual Smoke Test for Groq AI Integration.

Executes a single test completion request to Groq only when run directly from the CLI.
Never executed automatically by pytest. Never exposes tokens or secrets.

Usage:
  py backend/scripts/test_groq_ai.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv

# Load backend/.env if it exists
env_path = root_dir / "backend" / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from backend.app.ai.groq_provider import GroqAIProvider


def mask_key(key: str) -> str:
    """Masks secret API key for display."""
    if not key:
        return "<not set>"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def main() -> int:
    print("==================================================")
    print("            GROQ AI MANUAL SMOKE TEST             ")
    print("==================================================")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    print(f"Provider: Groq AI (OpenAI-compatible REST API)")
    print(f"Base URL: {base_url}")
    print(f"Model:    {model}")
    print(f"API Key:  {mask_key(api_key)}")
    print("--------------------------------------------------")

    if not api_key or api_key == "your_groq_api_key_placeholder":
        print("[ERROR] GROQ_API_KEY is not configured in backend/.env.")
        print("Please obtain your Groq API key from https://console.groq.com/keys")
        print("and add it to backend/.env as: GROQ_API_KEY=gsk_...")
        return 1

    print("Attempting minimal live completion request to Groq...")
    provider = GroqAIProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=15.0,
        max_retries=1,
    )

    test_messages = [
        {"role": "system", "content": "You are an assistant. Respond concisely in under 10 words."},
        {"role": "user", "content": "Ping test: please reply 'Groq AI connection successful'."},
    ]

    try:
        response = provider.generate_text(
            messages=test_messages,
            temperature=0.0,
            max_tokens=150,
        )
        print("\n[SUCCESS] Received response from Groq:")
        print(f"Response: '{response.text.strip()}'")
        print(f"Model:    {response.model}")
        print(
            f"Tokens:   {response.usage.total_tokens} total "
            f"({response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion)"
        )
        return 0

    except Exception as exc:
        print(f"\n[FAILED] Request to Groq encountered an error:")
        print(f"Error type: {exc.__class__.__name__}")
        print(f"Message:    {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
