"""Manual Smoke Test for Puter AI Integration.

Executes a single test request to Puter AI only when run directly from the CLI.
Never executed automatically by pytest. Never exposes tokens or secrets.

Usage:
  py backend/scripts/test_puter_ai.py
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

from backend.app.ai.puter_provider import PuterAIProvider
from backend.app.config import sanitize_db_url


def mask_token(token: str) -> str:
    """Masks secret token for display."""
    if not token:
        return "<not set>"
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


def main() -> int:
    print("==================================================")
    print("           PUTER AI MANUAL SMOKE TEST             ")
    print("==================================================")

    auth_token = os.getenv("PUTER_AUTH_TOKEN", "").strip()
    base_url = os.getenv("PUTER_BASE_URL", "https://api.puter.com/puterai/openai/v1/")
    model = os.getenv("PUTER_MODEL", "gpt-4o-mini")

    print(f"Provider: Puter AI (OpenAI-compatible REST gateway)")
    print(f"Base URL: {base_url}")
    print(f"Model:    {model}")
    print(f"Token:    {mask_token(auth_token)}")
    print("--------------------------------------------------")

    if not auth_token or auth_token == "your_puter_auth_token_placeholder":
        print("[ERROR] PUTER_AUTH_TOKEN is not configured in backend/.env.")
        print("Please obtain your Puter Auth Token from https://puter.com/dashboard")
        print("and add it to backend/.env as: PUTER_AUTH_TOKEN=your_token_here")
        return 1

    print("Attempting minimal live completion request to Puter AI...")
    provider = PuterAIProvider(
        auth_token=auth_token,
        base_url=base_url,
        model=model,
        timeout=15.0,
        max_retries=1,
    )

    test_messages = [
        {"role": "system", "content": "You are a test assistant. Keep responses under 10 words."},
        {"role": "user", "content": "Ping test: please reply 'Puter AI connection successful'."},
    ]

    try:
        response = provider.generate_text(
            messages=test_messages,
            temperature=0.0,
            max_tokens=30,
        )
        print("\n[SUCCESS] Received response from Puter AI:")
        print(f"Response: '{response.text.strip()}'")
        print(f"Model:    {response.model}")
        print(f"Tokens:   {response.usage.total_tokens} total ({response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion)")
        return 0

    except Exception as exc:
        print(f"\n[FAILED] Request to Puter AI encountered an error:")
        print(f"Error type: {exc.__class__.__name__}")
        print(f"Message:    {exc}")
        print("\nNote: If this host has network/firewall policies blocking api.puter.com,")
        print("or if the token has expired/insufficient balance, verify network & credentials.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
