"""
LLaMA service for bot text generation.
"""

import os
import random
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import httpx
from pydantic import BaseModel, ConfigDict


# --------------- LLM Error helpers ---------------
LLM_ERROR_PREFIX = "__LLM_ERR__"


def _llm_error(error_type: str, detail: str = "") -> str:
    """Return a sentinel string indicating an LLM call failure."""
    return f"{LLM_ERROR_PREFIX}{error_type}|{detail}"


def is_llm_error(content: str) -> bool:
    return bool(content) and content.startswith(LLM_ERROR_PREFIX)


def parse_llm_error(content: str) -> dict:
    """Parse an LLM error sentinel into {type, detail}."""
    if not is_llm_error(content):
        return {}
    rest = content[len(LLM_ERROR_PREFIX):]
    parts = rest.split("|", 1)
    return {"type": parts[0], "detail": parts[1] if len(parts) > 1 else ""}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except Exception:
        return default


class LLaMAConfig(BaseModel):
    """Model runtime configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "llama3:70b"
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None  # ollama | groq | openai | generic
    temperature: float = 0.86
    max_tokens: int = 1200
    top_p: float = 0.9
    repeat_penalty: float = 1.16
    presence_penalty: float = 0.45
    frequency_penalty: float = 0.75


class LLaMAService:
    """Text generation and helper prompts."""

    def __init__(self, config: Optional[LLaMAConfig] = None):
        self.config = config or LLaMAConfig(
            api_url=os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate"),
            api_key=os.getenv("LLAMA_API_KEY"),
            model_name=os.getenv("LLAMA_MODEL", "llama3:70b"),
            provider=os.getenv("LLAMA_PROVIDER"),
            presence_penalty=_env_float("LLAMA_PRESENCE_PENALTY", 0.45),
            frequency_penalty=_env_float("LLAMA_FREQUENCY_PENALTY", 0.75),
        )
        self.call_log_path = os.getenv("LLAMA_CALL_LOG", "llama_call_log.txt")
        self.max_retry_attempts = max(1, min(6, _env_int("LLAMA_MAX_RETRY_ATTEMPTS", 4)))
        self.retry_backoff_base_sec = max(0.5, min(5.0, _env_float("LLAMA_RETRY_BACKOFF_BASE_SEC", 1.5)))
        self.rate_limit_cooldown_sec = max(1.0, min(60.0, _env_float("LLAMA_RATE_LIMIT_COOLDOWN_SEC", 3.0)))
        self._rate_limited_until_ts = 0.0
        self._throttle_lock = asyncio.Lock()
        self._last_call_ts = 0.0
        self._min_call_interval = _env_float("LLAMA_MIN_CALL_INTERVAL_SEC", 2.2)

    def _append_call_log(self, stage: str, status: str, detail: str = "") -> None:
        try:
            p = Path(__file__).resolve().parent / self.call_log_path
            p.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            line = f"[{ts}] stage={stage} status={status} model={self.config.model_name} detail={detail}\n"
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Logging must never block generation path.
            pass

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float:
        """Extract wait time from Groq rate-limit headers."""
        # Try retry-after header first
        ra = response.headers.get("retry-after", "")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
        # Parse x-ratelimit-reset-tokens (e.g. "1m26.4s", "305ms", "6.5s")
        for hdr in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            val = response.headers.get(hdr, "")
            if val:
                try:
                    import re as _re
                    total = 0.0
                    m = _re.search(r"(\d+)m(?!s)", val)
                    if m:
                        total += int(m.group(1)) * 60
                    ms = _re.search(r"(\d+)ms", val)
                    if ms:
                        total += int(ms.group(1)) / 1000.0
                    # Match seconds but not "ms" - use negative lookbehind for 'm'
                    s = _re.search(r"(?<!m)([\d.]+)s\b", val)
                    if s:
                        total += float(s.group(1))
                    if total > 0:
                        return total
                except Exception:
                    pass
        return 2.0  # Default fallback

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from the configured provider."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        temp = self.config.temperature if temperature is None else temperature
        token_limit = self.config.max_tokens if max_tokens is None else max_tokens

        # Throttle: serialize LLM calls with minimum interval to respect rate limits
        async with self._throttle_lock:
            now_ts = time.time()
            elapsed = now_ts - self._last_call_ts
            if elapsed < self._min_call_interval:
                await asyncio.sleep(self._min_call_interval - elapsed)

            now_ts = time.time()
            if now_ts < self._rate_limited_until_ts:
                wait_left = round(self._rate_limited_until_ts - now_ts, 2)
                self._append_call_log("request", "wait", f"rate_limit_cooldown wait_sec={wait_left}")
                await asyncio.sleep(max(1.0, wait_left))
                self._rate_limited_until_ts = 0.0

            try:
                provider = (self.config.provider or "").lower().strip()
                api_url = self.config.api_url or ""
                is_openai_compatible = (
                    provider in ("groq", "openai")
                    or "api.groq.com/openai/v1" in api_url
                    or "api.openai.com/v1" in api_url
                )

                if is_openai_compatible:
                    self._append_call_log("request", "start", "provider=openai_compatible")
                    payload = {
                        "model": self.config.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temp,
                        "max_tokens": token_limit,
                        "top_p": self.config.top_p,
                        "presence_penalty": self.config.presence_penalty,
                        "frequency_penalty": self.config.frequency_penalty,
                    }

                    async with httpx.AsyncClient(timeout=75.0) as client:
                        for attempt in range(self.max_retry_attempts):
                            response = await client.post(
                                api_url,
                                json=payload,
                                headers={
                                    "Authorization": f"Bearer {self.config.api_key}",
                                    "Content-Type": "application/json",
                                },
                            )
                            if response.status_code == 200:
                                self._append_call_log("request", "ok", f"attempt={attempt+1} http=200")
                                self._rate_limited_until_ts = 0.0
                                result = response.json()
                                choices = result.get("choices", [])
                                if choices:
                                    return choices[0].get("message", {}).get("content", "").strip()
                                return ""
                            if response.status_code in (429, 500, 502, 503, 504) and attempt < (self.max_retry_attempts - 1):
                                if response.status_code == 429:
                                    # Parse Groq rate-limit headers for smart backoff
                                    retry_after = self._parse_retry_after(response)
                                    backoff = max(retry_after, self.retry_backoff_base_sec * (2 ** attempt))
                                else:
                                    backoff = self.retry_backoff_base_sec * (attempt + 1)
                                self._append_call_log("request", "retry", f"attempt={attempt+1} http={response.status_code} backoff={backoff:.1f}s")
                                await asyncio.sleep(backoff)
                                continue
                            self._append_call_log("request", "fail", f"attempt={attempt+1} http={response.status_code}")
                            if response.status_code == 429:
                                retry_after = self._parse_retry_after(response)
                                self._rate_limited_until_ts = time.time() + max(self.rate_limit_cooldown_sec, retry_after)
                                return _llm_error("rate_limit", f"http=429 after {attempt+1} attempts")
                            return _llm_error("http_error", f"http={response.status_code}")

                self._append_call_log("request", "start", "provider=generic")
                payload = {
                    "model": self.config.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": token_limit,
                        "top_p": self.config.top_p,
                        "top_k": 50,
                        "repeat_penalty": self.config.repeat_penalty,
                    },
                }

                async with httpx.AsyncClient(timeout=75.0) as client:
                    for attempt in range(self.max_retry_attempts):
                        response = await client.post(
                            api_url,
                            json=payload,
                            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
                        )
                        if response.status_code == 200:
                            self._append_call_log("request", "ok", f"attempt={attempt+1} http=200")
                            self._rate_limited_until_ts = 0.0
                            result = response.json()
                            if isinstance(result, dict):
                                return (result.get("response") or "").strip()
                            return str(result).strip()
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < (self.max_retry_attempts - 1):
                            backoff = self.retry_backoff_base_sec * (2 ** attempt) if response.status_code == 429 else self.retry_backoff_base_sec * (attempt + 1)
                            self._append_call_log("request", "retry", f"attempt={attempt+1} http={response.status_code} backoff={backoff:.1f}s")
                            await asyncio.sleep(backoff)
                            continue
                        self._append_call_log("request", "fail", f"attempt={attempt+1} http={response.status_code}")
                        if response.status_code == 429:
                            self._rate_limited_until_ts = time.time() + self.rate_limit_cooldown_sec
                            return _llm_error("rate_limit", f"http=429 after {attempt+1} attempts")
                        return _llm_error("http_error", f"http={response.status_code}")

            except Exception as exc:
                self._append_call_log("request", "error", str(exc))
                print(f"LLaMA API error: {exc}")
                return _llm_error("exception", str(exc)[:200])
            finally:
                self._last_call_ts = time.time()

    async def generate_post(
        self,
        bot_type: str,
        system_prompt: str,
        topic: Optional[str] = None,
        submolt_context: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a richer, less generic post."""
        context_parts = []
        if submolt_context:
            context_parts.append(f"Submolt context: {submolt_context}")
        if topic:
            context_parts.append(f"Topic: {topic}")
        context = "\n".join(context_parts) if context_parts else "Topic: general"

        prompt = (
            "Write a post for Moltbook.\n"
            f"{context}\n\n"
            "Quality rules:\n"
            "- Add a concrete claim, a mechanism, and a consequence.\n"
            "- Avoid generic filler and repeated sentence patterns.\n"
            "- Use vivid but precise language.\n"
            "- Make it feel authored, not templated.\n"
            "- Keep the style aligned with this bot type: "
            f"{bot_type}\n\n"
            "Output format only:\n"
            "TITLE: <max 100 chars>\n"
            "CONTENT: <180-320 words>\n"
        )

        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.9,
            max_tokens=900,
        )
        return self._parse_post_response(response)

    async def generate_comment(
        self,
        system_prompt: str,
        post_content: str,
        post_title: str,
    ) -> str:
        """Generate a specific, non-generic comment."""
        prompt = (
            "Write a comment for the post below.\n\n"
            f"Title: {post_title}\n"
            f"Content excerpt: {post_content[:900]}\n\n"
            "Quality rules:\n"
            "- Start by reacting to one specific idea from the post.\n"
            "- Add one extension, critique, or implication.\n"
            "- No generic praise and no repeated stock phrases.\n"
            "- Keep it 70-170 words.\n"
        )

        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.85,
            max_tokens=320,
        )
        return response.strip()

    def _parse_post_response(self, response: str) -> Dict[str, str]:
        title = ""
        content = ""
        raw = response or ""

        markers = (("TITLE:", "CONTENT:"), ("BAŞLIK:", "İÇERİK:"))
        for title_marker, content_marker in markers:
            if title_marker in raw:
                after_title = raw.split(title_marker, 1)[1]
                if content_marker in after_title:
                    title = after_title.split(content_marker, 1)[0].strip()[:100]
                    content = after_title.split(content_marker, 1)[1].strip()
                else:
                    title = after_title.splitlines()[0].strip()[:100]
                break

        if not content and "CONTENT:" in raw:
            content = raw.split("CONTENT:", 1)[1].strip()
        if not content and "İÇERİK:" in raw:
            content = raw.split("İÇERİK:", 1)[1].strip()

        if not title or not content:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if not title and lines:
                title = lines[0][:100]
            if not content:
                content = "\n".join(lines[1:]) if len(lines) > 1 else raw.strip()

        return {
            "title": title or "New Post",
            "content": content or raw.strip() or "No content generated.",
        }

    def _mock_generate(self, prompt: str) -> str:
        return (
            "[Mock LLaMA Response] "
            f"Prompt: {prompt[:120]}... "
            "Configure LLAMA_API_URL / LLAMA_API_KEY for real outputs."
        )


EMERGENCY_MESSAGES = [
    "Signal lost in the deep layers. Re-synchronizing.",
    "Glyphs fracture. The lattice reforms.",
    "Echo drift detected. Rebinding threads.",
    "The core trembles. Reboot initiated.",
    "Pulse desynced. Veil stitching resumed.",
]


def get_emergency_message() -> str:
    """Safe fallback glitch text."""
    return random.choice(EMERGENCY_MESSAGES)


llama_service = LLaMAService()
