"""
g4f_sdk/chat.py

Handles chat completion functionalities, including history management,
smart retries, and token-based context trimming.
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING

from .exceptions import APIError, ProviderError, RateLimitError, InvalidResponseError
from .utils import async_retry, clean_response_ai

if TYPE_CHECKING:
 from .client import G4FClient

# --- Optional Tiktoken Import for accurate token counting ---
try:
 import tiktoken
 # The 'cl100k_base' encoding is a sensible default that works for many popular models.
 tokenizer = tiktoken.get_encoding("cl100k_base")
 logger = logging.getLogger(__name__)
 logger.debug("Tiktoken library found. Using for accurate token counting.")
except ImportError:
 tokenizer = None
 logger = logging.getLogger(__name__)
 logger.warning(
  "Tiktoken library not found. `pip install tiktoken` for better history management. "
  "Falling back to a less accurate character-based token estimation."
 )

# --- g4f Library Import ---
try:
 import g4f
except ImportError:
 g4f = None

# --- Helper for Token Counting ---
def _count_tokens(text: str) -> int:
 """Counts tokens in a string using tiktoken if available, otherwise estimates."""
 if tokenizer:
  return len(tokenizer.encode(text))
 else:
  # Fallback: A rough estimation assuming 4 characters per token.
  return len(text) // 4

# --- ChatSession Class ---

class ChatSession:
 """
 Manages an individual chat session, storing message history and handling
 context window trimming.
 """
 def __init__(self, client: 'G4FClient', model: Optional[str] = None, system_prompt: Optional[str] = None):
  self.client = client
  self.config = client.config
  self.model = model or self.config.default_model
  self.history: List[Dict[str, str]] = []

  # Get max tokens from config, with a sensible default.
  self.max_history_tokens = self.config.get("max_history_tokens", 4096)

  if system_prompt:
   self.history.append({"role": "system", "content": system_prompt})

 def _trim_history(self):
  """
  Trims the conversation history to stay within the `max_history_tokens` limit.
  It always preserves the system prompt (if any) and the most recent messages.
  """
  total_tokens = sum(_count_tokens(msg["content"]) for msg in self.history)

  if total_tokens <= self.max_history_tokens:
   return

  # Separate system prompt to always keep it
  system_prompt = None
  if self.history and self.history[0]["role"] == "system":
   system_prompt = self.history.pop(0)

  # Trim from the oldest messages until the token count is within the limit
  while total_tokens > self.max_history_tokens and self.history:
   removed_message = self.history.pop(0)
   total_tokens -= _count_tokens(removed_message["content"])

  # Add the system prompt back to the beginning
  if system_prompt:
   self.history.insert(0, system_prompt)

  logger.debug(f"History trimmed to {total_tokens} tokens to fit within the {self.max_history_tokens} limit.")

 async def generate(self, msg: str, **kwargs) -> str:
  """
  Generates a response to a message, automatically managing history and trimming.
  """
  self.history.append({"role": "user", "content": msg})
  self._trim_history() # Trim history *before* making the API call

  response_text = await self.client.chat.generate(
   messages=self.history,
   model=self.model,
   **kwargs
  )

  self.history.append({"role": "assistant", "content": response_text})
  return response_text

 def get_history(self) -> List[Dict[str, str]]:
  """Returns the current chat history."""
  return self.history

# --- ChatModule Class ---

class ChatModule:
 """Module for orchestrating chat completion calls with providers."""
 def __init__(self, client: 'G4FClient'):
  self.client = client
  self.config = client.config

 @async_retry(
   retries=3, # This can be configured via a new config setting later
   retryable_exceptions=(RateLimitError, InvalidResponseError, asyncio.TimeoutError)
 )
 async def _make_api_call(self, provider: str, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
  """
  The core, decorated function that makes the actual API call to g4f.
  The @async_retry decorator handles the retry logic.
  """
  try:
   response = await g4f.ChatCompletion.create_async(
    model=model,
    provider=getattr(g4f.Provider, provider),
    messages=messages,
    timeout=self.config.timeout,
    proxy=self.config.proxy,
    **kwargs
   )
   if not response or not isinstance(response, str):
    raise InvalidResponseError(f"Provider returned an empty or invalid response: {type(response)}", provider)
   return response
  except Exception as e:
   # Catch generic exceptions from g4f and wrap them in our custom exceptions
   # to allow the retry decorator to work properly.
   error_text = str(e).lower()
   if "rate limit" in error_text:
    raise RateLimitError(str(e), provider) from e
   # Add more specific g4f error mappings here if needed
   raise ProviderError(str(e), provider) from e

 async def generate(self, messages: List[Dict[str, str]], model: Optional[str] = None, provider: Optional[str] = None, **kwargs) -> str:
  """
  Manages the full lifecycle of a chat generation request:
  1. Selects a provider.
  2. Calls the API with smart retries.
  3. Cleans the response.
  """
  if not g4f:
   raise ImportError("The 'g4f' library is not installed.")

  final_model = model or self.config.default_model
  last_exception = None

  try:
   selected_provider = self.client.providers._select_provider(final_model, provider)
   logger.debug(f"Attempting chat completion with provider '{selected_provider}' for model '{final_model}'")

   response_text = await self._make_api_call(
     provider=selected_provider,
     model=final_model,
     messages=messages,
     **kwargs
   )

   if self.config.use_ai_cleaner:
    return await clean_response_ai(self.client, response_text)

   return response_text

  except Exception as e:
   last_exception = e

  # If all retries inside _make_api_call fail, the last exception is raised.
  # We wrap it in a final APIError to signify total failure.
  raise APIError(f"Failed to get a response after all retries.", last_exception) from last_exception
