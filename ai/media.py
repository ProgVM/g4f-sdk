# ai/media.py

import asyncio
import logging
from typing import Optional, Union, Any, List, Literal
from pathlib import Path
import io

from g4f.models import flux, whisper_1, tts_1
from .config import _BaseHandler
from .models_info import get_model_provider_info

logger = logging.getLogger("g4f-sdk.media")

# --- Handler for Image Generation ---
class ImageHandler(_BaseHandler):
    """Handles image generation tasks."""
    async def generate(self, prompt: str, *, model=flux, provider: Optional[str] = None, **kwargs) -> Optional[str]:
        max_retries = self.config.get("max_retries")
        timeout = kwargs.get('timeout', self.config.get("timeout"))

        info = get_model_provider_info(model.name if hasattr(model, 'name') else str(model), provider)

        g4f_kwargs = {"model": model, "prompt": prompt, "provider": info.get('chosen_provider'), **kwargs}
        g4f_kwargs = {k: v for k, v in g4f_kwargs.items() if v is not None}

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(self.async_client.images.generate(**g4f_kwargs), timeout=timeout)
                return response.data[0].url
            except Exception as e:
                logger.warning(f"Image generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1: await asyncio.sleep(2)

        logger.error("Failed to generate image after all retries.")
        return None

# --- Handler for Audio ---
class AudioHandler(_BaseHandler):
    """Handles Text-to-Speech and Speech-to-Text tasks."""
    async def speech_to_text(self, file: Union[str, Path, bytes], *, model=whisper_1, provider: Optional[str] = None, **kwargs) -> Optional[str]:
        if isinstance(file, (str, Path)):
            with open(file, "rb") as audio_file:
                return await self._transcribe_call(model, audio_file, provider, **kwargs)
        elif isinstance(file, bytes):
            audio_file = io.BytesIO(file)
            return await self._transcribe_call(model, audio_file, provider, **kwargs)
        logger.error("Invalid file type for speech_to_text. Must be a path or bytes.")
        return None

    async def _transcribe_call(self, model, file_obj, provider, **kwargs):
        max_retries = self.config.get("max_retries")
        timeout = kwargs.get('timeout', self.config.get("timeout"))
        info = get_model_provider_info(model.name, provider)

        g4f_kwargs = {"model": model, "file": file_obj, "provider": info.get('chosen_provider'), **kwargs}

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(self.async_client.audio.transcriptions.create(**g4f_kwargs), timeout=timeout)
                return response.text
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1: await asyncio.sleep(2)
        return None

    async def text_to_speech(self, text: str, *, model=tts_1, provider: Optional[str] = None, **kwargs) -> Optional[bytes]:
        max_retries = self.config.get("max_retries")
        timeout = kwargs.get('timeout', self.config.get("timeout"))
        info = get_model_provider_info(model.name, provider)

        g4f_kwargs = {"model": model, "input": text, "provider": info.get('chosen_provider'), **kwargs}

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(self.async_client.audio.speech.create(**g4f_kwargs), timeout=timeout)
                return await response.aread()
            except Exception as e:
                logger.warning(f"TTS attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1: await asyncio.sleep(2)
        return None