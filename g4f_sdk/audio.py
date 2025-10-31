"""
g4f_sdk/audio.py

Handles audio transcription and synthesis with smart retries.
"""
import asyncio
import logging
from typing import Optional, Any, TYPE_CHECKING

from .exceptions import APIError, ProviderError, RateLimitError, InvalidResponseError
from .utils import async_retry

if TYPE_CHECKING:
    from .client import G4FClient

logger = logging.getLogger(__name__)

# --- g4f Library Import ---
try:
    import g4f
except ImportError:
    g4f = None

class AudioModule:
    """Module for audio tasks (transcription and synthesis)."""
    def __init__(self, client: 'G4FClient'):
        self.client = client
        self.config = client.config

    # --- Transcription Methods ---

    @async_retry(
        retries=3,
        retryable_exceptions=(RateLimitError, InvalidResponseError, asyncio.TimeoutError)
    )
    async def _make_transcribe_call(self, provider: str, model: str, audio_file: Any, **kwargs) -> str:
        """Core, decorated function for the g4f transcription API call."""
        try:
            response = await g4f.Stt.create_async(
                model=model,
                provider=getattr(g4f.Provider, provider),
                file=audio_file,
                timeout=self.config.timeout,
                **kwargs
            )
            if not response or not isinstance(response, str):
                raise InvalidResponseError("Provider returned an empty or invalid transcription.", provider)
            return response
        except Exception as e:
            error_text = str(e).lower()
            if "rate limit" in error_text:
                raise RateLimitError(str(e), provider) from e
            raise ProviderError(str(e), provider) from e

    async def transcribe(self, audio_path: str, model: Optional[str] = None, provider: Optional[str] = None, **kwargs) -> str:
        """Transcribes an audio file into text."""
        if not g4f:
            raise ImportError("The 'g4f' library is not installed.")

        final_model = model or "whisper"

        try:
            selected_provider = self.client.providers._select_provider(final_model, provider)
            logger.debug(f"Attempting transcription with provider '{selected_provider}' for model '{final_model}'")

            with open(audio_path, "rb") as audio_file:
                transcription_text = await self._make_transcribe_call(
                    provider=selected_provider,
                    model=final_model,
                    audio_file=audio_file,
                    **kwargs
                )
            return transcription_text
        except Exception as e:
            raise APIError(f"Failed to transcribe audio after all retries.", last_exception=e) from e

    # --- Text-to-Speech Methods ---

    @async_retry(
        retries=3,
        retryable_exceptions=(RateLimitError, InvalidResponseError, asyncio.TimeoutError)
    )
    async def _make_tts_call(self, provider: str, model: str, text: str, **kwargs) -> bytes:
        """Core, decorated function for the g4f text-to-speech API call."""
        try:
            response = await g4f.Speech.create_async(
                model=model,
                provider=getattr(g4f.Provider, provider),
                input=text,
                timeout=self.config.timeout,
                **kwargs
            )
            if not response or not isinstance(response, bytes):
                raise InvalidResponseError("Provider returned empty or invalid audio data.", provider)
            return response
        except Exception as e:
            error_text = str(e).lower()
            if "rate limit" in error_text:
                raise RateLimitError(str(e), provider) from e
            raise ProviderError(str(e), provider) from e

    async def text_to_speech(self, text: str, model: Optional[str] = None, provider: Optional[str] = None, **kwargs) -> bytes:
        """Converts text into speech audio."""
        if not g4f:
            raise ImportError("The 'g4f' library is not installed.")

        final_model = model or "tts-1"

        try:
            selected_provider = self.client.providers._select_provider(final_model, provider)
            logger.debug(f"Attempting text-to-speech with provider '{selected_provider}' for model '{final_model}'")

            audio_data = await self._make_tts_call(
                provider=selected_provider,
                model=final_model,
                text=text,
                **kwargs
            )
            return audio_data
        except Exception as e:
            raise APIError(f"Failed to synthesize speech after all retries.", last_exception=e) from e