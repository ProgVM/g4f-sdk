"""
g4f_sdk/images.py

Handles image generation functionalities with smart retries.
"""
import asyncio
import logging
from typing import Optional, List, Any, TYPE_CHECKING

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

class ImageModule:
    """Module for generating images, using a retry mechanism for reliability."""
    def __init__(self, client: 'G4FClient'):
        self.client = client
        self.config = client.config

    @async_retry(
        retries=3,
        retryable_exceptions=(RateLimitError, InvalidResponseError, asyncio.TimeoutError)
    )
    async def _make_api_call(self, provider: str, model: str, prompt: str, **kwargs) -> List[str]:
        """The core, decorated function for the g4f image generation API call."""
        try:
            response = await g4f.Image.create_async(
                model=model,
                provider=getattr(g4f.Provider, provider),
                prompt=prompt,
                proxy=self.config.proxy,
                timeout=self.config.timeout,
                **kwargs
            )
            if not response or not isinstance(response, list) or not response[0]:
                raise InvalidResponseError("Provider returned an empty or invalid list of images.", provider)
            return response
        except Exception as e:
            error_text = str(e).lower()
            if "rate limit" in error_text:
                raise RateLimitError(str(e), provider) from e
            raise ProviderError(str(e), provider) from e

    async def generate(self, prompt: str, model: Optional[str] = None, provider: Optional[str] = None, **kwargs) -> str:
        """
        Manages the image generation request: selects a provider and calls the API.
        Returns the URL of the first generated image.
        """
        if not g4f:
            raise ImportError("The 'g4f' library is not installed.")

        final_model = model or "dall-e-3" # A reasonable default for images

        try:
            selected_provider = self.client.providers._select_provider(final_model, provider)
            logger.debug(f"Attempting image generation with provider '{selected_provider}' for model '{final_model}'")

            image_urls = await self._make_api_call(
                provider=selected_provider,
                model=final_model,
                prompt=prompt,
                **kwargs
            )

            # Return the first image URL from the list
            return image_urls[0]

        except Exception as e:
            # Re-raise as a final APIError after all retries have failed
            raise APIError(f"Failed to generate image after all retries.", last_exception=e) from e
