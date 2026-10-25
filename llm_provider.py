"""
LLM Provider module for Financial News Sentiment + RAG Analyst.
Implements an abstract LLM provider interface with a primary Groq Cloud API provider
and a Hugging Face Inference API fallback provider. Handles network errors, missing keys,
and rate limits gracefully without exposing secrets or crashing the application.
"""

import os
import abc
import logging
from typing import Optional, Dict, Any
import requests

from config import config, logger


class LLMProviderException(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMProvider(abc.ABC):
    """Abstract Base Class for LLM providers."""

    @abc.abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Generates a text completion response given system and user prompts.

        Parameters:
            system_prompt: System role instructions for the model.
            user_prompt: User query and context.
            temperature: Sampling temperature (default 0.0 for deterministic output).
            max_tokens: Maximum tokens in response.

        Returns:
            Generated response string.
        """
        pass


class GroqLLMProvider(LLMProvider):
    """
    Primary LLM provider using Groq Cloud API.
    Model: configurable via GROQ_MODEL environment variable (default: qwen/qwen3.8-27b).
    """

    GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", None)
        self.model = model or os.getenv("GROQ_MODEL") or getattr(config, "GROQ_MODEL", "qwen/qwen3.8-27b")
        self.timeout = timeout or getattr(config, "LLM_TIMEOUT", 20)

    def is_available(self) -> bool:
        """Checks if API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not self.is_available():
            raise LLMProviderException("Groq API key not found in environment (GROQ_API_KEY).")

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(f"Invoking Groq API model '{self.model}'...")
        try:
            response = requests.post(
                self.GROQ_ENDPOINT, headers=headers, json=payload, timeout=self.timeout
            )
            if response.status_code == 401:
                raise LLMProviderException("Groq API authentication failure: Invalid or expired API key.")
            elif response.status_code == 429:
                raise LLMProviderException("Groq API rate limit exceeded.")
            elif response.status_code in [400, 404]:
                try:
                    err_json = response.json().get("error", {})
                    err_code = err_json.get("code", "")
                    err_msg = err_json.get("message", response.text[:200])
                except Exception:
                    err_code = ""
                    err_msg = response.text[:200]
                if (
                    "decommissioned" in err_msg.lower()
                    or "not_found" in err_code.lower()
                    or "not found" in err_msg.lower()
                ):
                    raise LLMProviderException(
                        f"Groq unsupported model error: Model '{self.model}' is not recognized or has been decommissioned by Groq."
                    )
                raise LLMProviderException(f"Groq API error ({response.status_code}): {err_msg}")
            elif response.status_code != 200:
                raise LLMProviderException(
                    f"Groq API HTTP error {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderException("Groq API returned response with no choices.")

            answer = choices[0].get("message", {}).get("content", "").strip()
            return answer

        except requests.Timeout:
            logger.warning(f"Groq API request timed out after {self.timeout}s.")
            raise LLMProviderException(f"Groq API request timed out after {self.timeout}s.")
        except requests.RequestException as e:
            logger.warning(f"Groq API network call failed: {e}")
            raise LLMProviderException(f"Groq API connection error: {e}")


class HuggingFaceLLMProvider(LLMProvider):
    """
    Fallback LLM provider using Hugging Face Inference API.
    Model: meta-llama/Llama-3.1-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.2.
    """

    HF_API_BASE = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        token: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        if token is not None:
            self.token = token
        else:
            self.token = (
                os.getenv("HF_TOKEN")
                or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                or getattr(config, "HF_TOKEN", None)
            )
        self.model = model or os.getenv("HF_MODEL") or getattr(config, "HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.timeout = timeout or getattr(config, "LLM_TIMEOUT", 25)

    def is_available(self) -> bool:
        """Checks if Hugging Face token is present."""
        return bool(self.token and self.token.strip())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not self.is_available():
            raise LLMProviderException(
                "Hugging Face token not found in environment (HF_TOKEN / HUGGINGFACEHUB_API_TOKEN)."
            )

        endpoint = f"{self.HF_API_BASE}/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.token.strip()}",
            "Content-Type": "application/json",
        }

        prompt_combined = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        payload = {
            "inputs": prompt_combined,
            "parameters": {
                "temperature": max(0.01, temperature),
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
        }

        logger.info(f"Invoking HuggingFace Inference API model '{self.model}'...")
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            if response.status_code == 401:
                raise LLMProviderException("HuggingFace API authentication failure: Invalid or expired token.")
            elif response.status_code == 429:
                raise LLMProviderException("HuggingFace API rate limit exceeded.")
            elif response.status_code != 200:
                raise LLMProviderException(
                    f"HuggingFace API HTTP error {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                generated_text = data[0].get("generated_text", "").strip()
                return generated_text
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            else:
                raise LLMProviderException(f"Unexpected HF response format: {str(data)[:150]}")

        except requests.Timeout:
            logger.warning(f"HuggingFace API request timed out after {self.timeout}s.")
            raise LLMProviderException(f"HuggingFace API request timed out after {self.timeout}s.")
        except requests.RequestException as e:
            logger.warning(f"HuggingFace API network call failed: {e}")
            raise LLMProviderException(f"HuggingFace API connection error: {e}")


class FallbackLLMProvider(LLMProvider):
    """
    Orchestrated LLM provider that attempts Groq first and falls back to Hugging Face.
    Returns a controlled error message if both providers are unavailable or fail.
    """

    CONTROLLED_FALLBACK_MESSAGE = (
        "I’m unable to generate an analyst response right now because the "
        "language model providers are unavailable."
    )

    def __init__(
        self,
        groq_provider: Optional[GroqLLMProvider] = None,
        hf_provider: Optional[HuggingFaceLLMProvider] = None,
    ):
        self.groq_provider = groq_provider or GroqLLMProvider()
        self.hf_provider = hf_provider or HuggingFaceLLMProvider()
        self.last_active_provider: Optional[str] = None
        self.last_model_used: Optional[str] = None
        self.last_error: Optional[str] = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Attempts Groq API first; falls back to HuggingFace API if Groq fails.
        Returns a controlled string message if both fail.
        """
        self.last_active_provider = None
        self.last_model_used = None
        self.last_error = None

        # Attempt 1: Groq API
        if self.groq_provider.is_available():
            try:
                logger.info(f"Attempting primary LLM provider (Groq: {self.groq_provider.model})...")
                response = self.groq_provider.generate(
                    system_prompt, user_prompt, temperature, max_tokens
                )
                if response and response.strip():
                    self.last_active_provider = f"Groq ({self.groq_provider.model})"
                    self.last_model_used = self.groq_provider.model
                    return response
            except LLMProviderException as e:
                self.last_error = str(e)
                logger.warning(f"Primary provider (Groq) failed: {e}. Switching to fallback...")
        else:
            logger.info("Groq API key not configured. Skipping primary provider.")

        # Attempt 2: HuggingFace API Fallback
        if self.hf_provider.is_available():
            try:
                logger.info(f"Attempting fallback LLM provider (HuggingFace: {self.hf_provider.model})...")
                response = self.hf_provider.generate(
                    system_prompt, user_prompt, temperature, max_tokens
                )
                if response and response.strip():
                    self.last_active_provider = f"HuggingFace ({self.hf_provider.model})"
                    self.last_model_used = self.hf_provider.model
                    return response
            except LLMProviderException as e:
                self.last_error = str(e)
                logger.warning(f"Fallback provider (HuggingFace) failed: {e}.")
        else:
            logger.info("HuggingFace token not configured. Skipping fallback provider.")

        # Both providers failed or unconfigured: return controlled message
        logger.warning("Both LLM providers failed or unconfigured. Returning controlled response.")
        return self.CONTROLLED_FALLBACK_MESSAGE

    def get_health_status(self) -> Dict[str, Any]:
        """
        Returns health and configuration diagnostics for LLM providers.
        Safe for display: never exposes keys or tokens.
        """
        groq_ready = self.groq_provider.is_available()
        hf_ready = self.hf_provider.is_available()
        if groq_ready:
            active_provider = f"Groq Cloud ({self.groq_provider.model})"
            mode = "LLM AVAILABLE"
            explanation = f"Connected to Groq Cloud API ({self.groq_provider.model}) for natural-language analysis."
        elif hf_ready:
            active_provider = f"Hugging Face Hub ({self.hf_provider.model})"
            mode = "LLM FALLBACK"
            explanation = f"Connected to Hugging Face Inference API fallback ({self.hf_provider.model})."
        else:
            active_provider = "Factual Deterministic Fallback"
            mode = "LLM UNAVAILABLE"
            explanation = "No external LLM provider configured. Answers are deterministically grounded and synthesized directly from SQLite/Chroma evidence."

        return {
            "mode": mode,
            "active_provider": active_provider,
            "explanation": explanation,
            "groq_configured": groq_ready,
            "hf_configured": hf_ready,
            "groq_model": self.groq_provider.model,
            "hf_model": self.hf_provider.model,
            "last_active_provider": self.last_active_provider,
            "last_error": self.last_error,
        }
