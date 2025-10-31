"""
g4f_sdk/client.py

The main client class that orchestrates all SDK functionalities.
"""
from typing import Optional, List, Dict, Any

from .config import Config
from .providers import ProviderManager
from .chat import ChatModule, ChatSession
from .images import ImageModule
from .audio import AudioModule
from . import utils

class G4FClient:
 """
 The main entry point for the g4f-sdk.

 This client provides a unified interface for chat, image generation, and
 audio tasks, backed by a resilient and configurable architecture.
 """
 def __init__(self, config_path: Optional[str] = None, **kwargs: Any):
  # 1. Initialize configuration
  self.config = Config(config_path, **kwargs)

  # 2. Setup logging based on config
  utils.setup_logging(self.config.log_level)

  # 3. Initialize core modules, passing a reference to this client instance
  self.providers = ProviderManager(self.config)
  self.chat = ChatModule(self)
  self.images = ImageModule(self)
  self.audio = AudioModule(self)
  self.utils = utils

 def get_working_providers(self) -> List[str]:
  """
  Returns a list of currently working provider names.
  Note: This list is cached based on `provider_cache_ttl`.
  """
  self.providers._ensure_cache()
  return list(self.providers._providers_info.keys())

 def new_chat(self, model: Optional[str] = None, system_prompt: Optional[str] = None) -> ChatSession:
  """
  Creates a new, isolated chat session with its own history.

  :param model: The model to use for this session (e.g., "gpt-4o"). Overrides client's default.
  :param system_prompt: An initial prompt to set the AI's behavior.
  :return: A ChatSession object for interactive conversations.
  """
  return ChatSession(self, model, system_prompt)

 async def generate_image(self, prompt: str, **kwargs: Any) -> str:
  """
  Generates an image from a text prompt.

  :param prompt: The text description of the image to generate.
  :param kwargs: Additional arguments for the API call (e.g., model, provider).
  :return: A URL string of the generated image.
  :raises APIError: If the image could not be generated after all retries.
  """
  return await self.images.generate(prompt, **kwargs)

 async def transcribe_audio(self, audio_path: str, **kwargs: Any) -> str:
  """
  Transcribes an audio file into text using a speech-to-text model.

  :param audio_path: The local file path to the audio file.
  :param kwargs: Additional arguments for the API call (e.g., model, provider).
  :return: The recognized text from the audio.
  :raises APIError: If transcription fails after all retries.
  """
  return await self.audio.transcribe(audio_path, **kwargs)

 async def text_to_speech(self, text: str, **kwargs: Any) -> bytes:
  """
  Converts text into speech audio.

  :param text: The text to synthesize.
  :param kwargs: Additional arguments for the API call (e.g., model, provider).
  :return: The audio data as bytes.
  :raises APIError: If synthesis fails after all retries.
  """
  return await self.audio.text_to_speech(text, **kwargs)