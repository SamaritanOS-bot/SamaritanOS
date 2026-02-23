"""
Moltbook claim durumunu kontrol eder.
"""

import os
from pathlib import Path
import httpx
from dotenv import load_dotenv


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def main() -> int:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    base = _env("MOLTBOOK_API_BASE", "https://www.moltbook.com/api/v1").rstrip("/")
    api_key = _env("MOLTBOOK_API_KEY")
    if not api_key:
        print("MOLTBOOK_API_KEY tanımlı değil.")
        return 1

    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{base}/agents/status", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        print("Claim status:", data.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
