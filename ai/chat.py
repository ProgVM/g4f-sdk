# ai/chat.py

import asyncio
import logging
import re
import json
from typing import Optional, List, Union, Literal, Any, AsyncGenerator
import tiktoken

from g4f.client import Client as g4f_client
from g4f.models import gemini_2_5_flash
from .config import _BaseHandler, Config
from .models_info import get_model_provider_info
from ai import _get_g4f_provider_class

logger = logging.getLogger("g4f-sdk.chat")

# --- Handler for Text and Vision Models ---
class ChatHandler(_BaseHandler):
 """Handles chat completions, context management, and vision tasks."""
 def __init__(self, config: Config, model=gemini_2_5_flash, context: Optional[List[dict]] = None):
  super().__init__(config)
  self.model = model
  self.msgs = context if context else []
  self._update_model_info()


 def _update_model_info(self, provider_ref: Optional[Union[str, type]] = None):
  """Updates internal model/provider info and configuration."""


  provider_name = provider_ref
  if provider_name and not isinstance(provider_name, str):
   provider_name = provider_name.__name__

  model_name = self.model.name if hasattr(self.model, 'name') else str(self.model)

  self.info = get_model_provider_info(model_name, provider_name)

  self.provider = self.info.get('chosen_provider')
  self.max_tokens = self.info.get('max_tokens', 8192)

  self.use_char_limit = 'pollinations' in (self.provider or "").lower()
  self.max_length = self.info.get('max_chars', 30000) if self.use_char_limit else self.max_tokens

  logger.info(f"ChatHandler configured for {model_name} via {self.provider}. Max length: {self.max_length}")

 def get_context(self) -> List[dict]: return self.msgs
 def set_context(self, messages: List[dict]): self.msgs = messages
 def clear_context(self, keep_system_prompt: bool = True):
  """Clears the chat history, optionally keeping the system prompt."""
  if keep_system_prompt and self.msgs and self.msgs[0].get("role") == "system":
   self.msgs = [self.msgs[0]]
  else:
   self.msgs = []

 def _count_length(self, messages: List[dict]) -> int:
  """Calculates the length of messages, using tokens or characters based on provider."""
  if self.use_char_limit:
   return sum(len(str(msg.get("content", ""))) for msg in messages)
  # Using tiktoken for accurate token count
  if self.encoding:
   # Note: This is an approximation since message structure also takes tokens.
   total_tokens = 0
   for msg in messages:
    # Content can be a list (for vision) or a string
    content = msg.get("content", "")
    if isinstance(content, str):
     total_tokens += len(self.encoding.encode(content))
    elif isinstance(content, list):
     # Approximate count for multimodal/vision messages
     for item in content:
      if item.get("type") == "text":
       total_tokens += len(self.encoding.encode(item.get("text", "")))
      # No easy way to count image tokens without model specific logic, so we skip it.
   return total_tokens

  # Fallback to character count if encoding is unavailable
  return sum(len(str(msg.get("content", ""))) for msg in messages)

 def _trim_history(self) -> bool:
  """
  Trims the message history based on context_reduction_factor if it exceeds max_length.
  Returns True if successful, False if trimming logic needs adaptation.
  """
  # FIX: Corrected indentation for reduction_factor
  reduction_factor = self.config.get("context_reduction_factor")
  if self._count_length(self.msgs) <= self.max_length: return True

  system_prompt = self.msgs[0] if self.msgs and self.msgs[0].get("role") == "system" else None
  other_msgs = self.msgs[1:] if system_prompt else self.msgs

  if reduction_factor >= 1.0:
   # Keep last N messages
   other_msgs = other_msgs[-int(reduction_factor):]
  else:
   # Trim to X percent of max length
   target_length = int(self.max_length * reduction_factor)
   while self._count_length(([system_prompt] if system_prompt else []) + other_msgs) > target_length and other_msgs:
    other_msgs.pop(0)

  self.msgs = ([system_prompt] if system_prompt else []) + other_msgs
  return self._count_length(self.msgs) <= self.max_length

 async def _clean_response(
  self,
  response_content: str,
  model: Optional[Any] = None,
  provider: Optional[str] = None,
  **kwargs
 ) -> str:
  """
  Uses a secondary AI model to detect and remove provider-inserted ads or artifacts.

  :param response_content: The text content to clean.
  :param model: The g4f model to use for the cleaning task (defaults to gemini_2_5_flash).
  :returns: The cleaned response content.
  """
  try:
   # Use a fast, reliable model for the JSON cleaning task
   ad_cleaning_model = model or gemini_2_5_flash

   # Use the current provider if specified, otherwise rely on g4f's internal selection
   cleaning_provider = provider if provider != self.provider else None

   system_prompt = """
   You are an expert in detecting ads and provider artifacts from the g4f-provider. Ads usually appear at the end of the response in a non-standard format (e.g., sponsor blocks, footers, or external links not requested by the user).

   Analyze the full user-provided text. Distinguish provider ads (inserted automatically) from text written by the AI based on the user's request.

   Return JSON: {"has_ad": true/false, "cleaned_text": "corrected text"}.

   Do not add any extra text outside the JSON structure.
   """

   ad_response = await self.async_client.chat.completions.create(
    model=ad_cleaning_model,
    messages=[\
     {"role": "system", "content": system_prompt},\
     {"role": "user", "content": f"Check this text for ads:\n{response_content}"}\
    ],
    provider=cleaning_provider,
    timeout=15, # Use a short timeout for this secondary task
    **kwargs # Pass other relevant kwargs like temperature
   )

   ad_content = ad_response.choices[0].message.content
   ad_result = json.loads(ad_content.strip())

   if ad_result.get("has_ad", False):
    return ad_result.get("cleaned_text", response_content).strip()

   return response_content.strip()

  except json.JSONDecodeError:
   logger.warning("Ad cleaning model failed to return valid JSON. Returning original text.")
   return response_content.strip()
  except Exception as e:
   logger.warning(f"G4F-SDK Ad Cleaning Failed (Falling back to original response): {e}")
   return response_content.strip()

 async def generate(self, msg: Optional[str] = None, *, provider: Optional[str] = None, web_search: bool = False, images: Optional[List[Any]] = None, return_format: Literal['tuple', 'content'] = 'tuple', **kwargs) -> Any:
  """
  Generates a non-streaming chat response with retries and adaptive context trimming.

  :param msg: The user's message.
  :param provider: Overrides the default provider for this call.
  :param web_search: Enable web search/browsing if supported by the model/provider.
  :param images: A list of image data (e.g., base64 or URL structure) for vision models.
  :param return_format: 'tuple' (content, history) or 'content' (content only).
  :returns: The generated content or a tuple of content and history.
  """

  # 1. Update/Check Model Capabilities
  if provider and provider != self.provider: self._update_model_info(provider)
  if web_search and not self.info.get('supports_web_search'):
   logger.warning(f"Web search not supported by '{self.provider}'. Ignoring request."); web_search = False
  if images and not self.info.get('supports_vision'):
   logger.warning(f"Vision not supported by '{self.provider}'. Ignoring images."); images = None

  max_retries = self.config.get("max_retries")
  timeout = kwargs.get('timeout', self.config.get("timeout"))

  # 2. Append user message to history
  if msg: 
   user_msg = {"role": "user", "content": msg}
   if images: user_msg["images"] = images # ADDED: images support for history
   self.msgs.append(user_msg)

  # 3. Resilient Request Loop
  g4f_kwargs = {"model": self.model, "messages": self.msgs, "stream": False, "provider": self.provider, "web_search": web_search, **kwargs}

  # NOTE: api_key and other custom arguments are kept in kwargs for provider specific use.
  g4f_kwargs = {k: v for k, v in g4f_kwargs.items() if v is not None}

  for attempt in range(max_retries):
   try:
    if not self._trim_history():
     # Adaptive max_length reduction if trimming fails (provider gave a bad limit)
     old_max = self.max_length
     self.max_length = int(self.max_length * 0.8)
     logger.warning(f"Trimming failed, adaptively reducing max_length from {old_max} to {self.max_length}")
     if self.max_length < 500: return None
     continue

    response = await asyncio.wait_for(self.async_client.chat.completions.create(**g4f_kwargs), timeout=timeout)

    # In non-streaming mode, content is always here
    content = response.choices[0].message.content

    if content:
     self.msgs.append({"role": "assistant", "content": content})
     if kwargs.get('clean_ad'):
      # Using the new clean_response with parameters
      content = await self._clean_response(content, **kwargs)
      self.msgs[-1]['content'] = content

     if return_format == 'content': return content
     return content, self.msgs
    return None

   except asyncio.TimeoutError:
    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Request timed out.")
   except Exception as e:
    error_str = str(e).lower()
    # Provider specific limit handling
    if "maximum context length" in error_str or "exceeds maximum length" in error_str:
     match = re.search(r'(\d+)\s*tokens', error_str)
     if match:
      provider_max = int(match.group(1))
      self.max_length = provider_max - 500
      logger.info(f"Provider limit detected. Adaptively adjusting max_length to {self.max_length}")
      continue
    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

   if attempt < max_retries - 1: await asyncio.sleep(2)

  logger.error("Failed to get response after all retries.")
  return None

 async def stream_generate(
  self,
  msg: str,
  system_prompt: Optional[str] = None,
  model: Optional[Any] = None,
  images: Optional[List[Any]] = None,
  web_search: bool = False,
  **kwargs
 ) -> AsyncGenerator[str, None]:
  """
  Generates a response chunk by chunk with resilience (automatic retries).

  :param msg: The user's message.
  :param images: A list of image data for vision models.
  :returns: An AsyncGenerator yielding string chunks of the response.
  """
  current_model = model or self.model
  max_retries = self.config.get("max_retries")
  timeout = kwargs.get('timeout', self.config.get("timeout"))

  full_response_content = ""
  original_msg = msg

  for attempt in range(max_retries):
   try:
    # 1. Prepare context for current attempt
    temp_history = self.msgs.copy()
    final_msg = original_msg

    # If this is a retry, modify history and message to request continuation
    if full_response_content:
     # Note: We append images to the user message in the retry history for multimodal consistency
     temp_history.append({"role": "user", "content": original_msg, "images": images})
     temp_history.append({"role": "assistant", "content": full_response_content})
     final_msg = "[CONTINUE THE RESPONSE WITHOUT REPEATING THE PREVIOUS TEXT]"

    # Add current user message to temp_history for trimming/sending
    user_message_to_send = {"role": "user", "content": final_msg}
    if images: user_message_to_send["images"] = images # Include images if present
    temp_history.append(user_message_to_send)

    # 2. Trim history based on temp_history (using _trim_history logic)
    original_msgs = self.msgs.copy()
    self.msgs= temp_history

    if not self._trim_history():
     logger.warning(f"Trimming failed in stream_generate. Stopping retry.")
     self.msgs = original_msgs
     return

    messages_to_send = self.msgs.copy()
    self.msgs = original_msgs

    # 3. Call g4f with stream=True
    g4f_kwargs = {
     "model": current_model,
     "messages": messages_to_send,
     "stream": True,
     "provider": self.provider,
     "web_search": web_search,
     **kwargs
    }

    # NOTE: api_key and other custom arguments are kept in kwargs for provider specific use.
    g4f_kwargs = {k: v for k, v in g4f_kwargs.items() if v is not None}

    response_stream = await asyncio.wait_for(
     self.async_client.chat.completions.create(**g4f_kwargs),
     timeout=timeout
    )

    # 4. Iterate over the stream and yield chunks
    async for chunk in response_stream:
     if chunk.choices[0].delta.content:
      new_chunk = chunk.choices[0].delta.content

      yield new_chunk
      full_response_content += new_chunk

    # 5. Success: Update main history and exit
    user_message_final = {"role": "user", "content": original_msg}
    if images: user_message_final["images"] = images

    self.msgs.append(user_message_final)
    self.msgs.append({"role": "assistant", "content": full_response_content})
    return

   except asyncio.TimeoutError:
    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Stream request timed out.")
   except Exception as e:
    # Log other errors and retry
    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed in stream: {e}")

   if attempt < max_retries - 1: await asyncio.sleep(2)

  logger.error("Failed to get streaming response after all retries.")
  return