from __future__ import annotations

import uuid

import httpx

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"
OLLAMA_TIMEOUT = 60.0


class OllamaConnectionError(Exception):
    pass


def _new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


class OllamaCopilotClient:
    def __init__(self, base_url: str = OLLAMA_BASE, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def ask_copilot(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict] | None = None,
    ) -> tuple[str, str]:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
        except httpx.ConnectError:
            raise OllamaConnectionError(
                "Cannot connect to Ollama. Start the Ollama application."
            )
        except httpx.TimeoutException:
            raise OllamaConnectionError(
                "Ollama request timed out. Ensure the model is loaded."
            )
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(
                f"Ollama returned an error (HTTP {e.response.status_code})."
            )

        correlation_id = _new_correlation_id()
        return answer, correlation_id
