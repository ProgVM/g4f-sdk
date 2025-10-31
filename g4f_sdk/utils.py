"""
g4f_sdk/utils.py

Core utilities for the SDK, including a smart retry decorator,
logging setup, and response cleaning functions.
"""
import asyncio
import logging
import re
from functools import wraps
from typing import Callable, Any, Coroutine, Tuple, Type, TYPE_CHECKING

from .exceptions import RateLimitError, InvalidResponseError

if TYPE_CHECKING:
  from .client import G4FClient

# Use a dedicated logger for utilities
logger = logging.getLogger(__name__)

# --- Logging Setup ---

def setup_logging(level: str = "INFO"):
  """
  Configures basic logging for the SDK.
  Avoids adding duplicate handlers if logging is already configured.
  """
  log_level = getattr(logging, level.upper(), logging.INFO)
  root_logger = logging.getLogger("g4f_sdk")

  # Check if handlers are already configured for our logger to avoid duplication
  if not root_logger.handlers:
    root_logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.propagate = False # Prevent passing messages to the root logger

# --- Smart Retry Decorator ---

def async_retry(
  retries: int = 3,
  delay: float = 2.0,
  backoff_factor: float = 2.0,
  retryable_exceptions: Tuple[Type[Exception], ...] = (
    RateLimitError,
    InvalidResponseError,
    asyncio.TimeoutError,
    # Add other temporary network or provider errors here
  )
) -> Callable[..., Coroutine[Any, Any, Any]]:
  """
  A decorator for retrying an async function with exponential backoff.

  It only retries on exceptions specified in `retryable_exceptions`.
  All other exceptions are raised immediately.
  """
  def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
      current_delay = delay
      for attempt in range(retries + 1):
        try:
          return await func(*args, **kwargs)
        except retryable_exceptions as e:
          if attempt == retries:
            logger.error(
              f"Function '{func.__name__}' failed after {retries + 1} attempts. Last error: {e}"
            )
            raise e from e

          logger.warning(
            f"Function '{func.__name__}' failed with a retryable error (Attempt {attempt + 1}/{retries + 1}): {e}. "
            f"Retrying in {current_delay:.2f} seconds..."
          )
          await asyncio.sleep(current_delay)
          current_delay *= backoff_factor
        except Exception as e:
          # Non-retryable exceptions are raised immediately
          logger.error(
            f"Function '{func.__name__}' failed with a non-retryable error: {e}"
          )
          raise e
    return wrapper
  return decorator

# --- Response Cleaning ---

def _clean_response_programmatic(text: str) -> str:
  """
  Performs basic, rule-based cleaning of the response text.
  This serves as a fallback if the AI cleaner is disabled or fails.
  """
  if not text:
    return ""

  # 1. Remove common disclaimer patterns
  patterns = [
    r"as an ai language model,\s*i cannot.*",
    r"i am not able to.*",
    r"i'm just an ai and do not have.*",
    r"disclaimer:.*",
    # Add more specific provider artifacts here as they are discovered
  ]
  cleaned_text = text
  for pattern in patterns:
    cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL).strip()

  # 2. Normalize whitespace
  cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text) # Replace multiple spaces with one
  cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text) # Replace multiple newlines with two

  return cleaned_text.strip()

async def clean_response_ai(client: 'G4FClient', text_to_clean: str) -> str:
  """
  Uses an AI call to clean the response from provider artifacts.
  Falls back to programmatic cleaning on failure.
  """
  if not text_to_clean:
    return ""

  system_prompt = (
    "You are a text cleaning expert. Your task is to remove any "
    "provider-specific artifacts, ads, disclaimers, or metadata from the given text. "
    "Return only the clean, core message that the user requested. Do not add any "
    "of your own commentary or introductions. Just return the cleaned text."
  )

  try:
    # Use a temporary, isolated chat session for the cleaning task
    # to not interfere with the user's main chat history.
    temp_chat_session = client.new_chat(
      system_prompt=system_prompt,
      model=client.config.default_model # Use a reliable default model for this task
    )
    cleaned_text = await temp_chat_session.generate(text_to_clean)
    return cleaned_text
  except Exception as e:
    logger.warning(f"AI response cleaning failed: {e}. Falling back to programmatic cleaning.")
    # Fallback to the rule-based cleaner if the AI call fails
    return _clean_response_programmatic(text_to_clean)
