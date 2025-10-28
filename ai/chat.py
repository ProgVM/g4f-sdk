# ai/chat.py

import asyncio
import logging
import re
from typing import Optional, List, Union, Literal, Any
import tiktoken

from g4f.models import gpt_4o
from .config import _BaseHandler, Config
from .models_info import get_model_provider_info

logger = logging.getLogger("g4f-sdk.chat")

# --- Handler for Text and Vision Models ---
class ChatHandler(_BaseHandler):
    """Handles chat completions, context management, and vision tasks."""
    def __init__(self, config: Config, model=gpt_4o, context: Optional[List[dict]] = None):
        super().__init__(config)
        self.model = model
        self.msgs = context if context else []
        self._update_model_info()

    def _update_model_info(self, provider_name: Optional[str] = None):
        model_name = self.model.name if hasattr(self.model, 'name') else str(self.model)
        self.info = get_model_provider_info(model_name, provider_name)

        self.provider = self.info.get('chosen_provider')
        self.max_tokens = self.info.get('max_tokens', 8192)

        self.use_char_limit = 'pollinations' in (self.provider or "").lower()
        self.max_length = self.info.get('max_chars', 30000) if self.use_char_limit else self.max_tokens

        if not self.use_char_limit:
            # We assume cl100k_base for most modern models (GPT/Claude/Gemini)
            self.encoding = tiktoken.get_encoding("cl100k_base") 

        logger.info(f"ChatHandler configured for {model_name} via {self.provider}. Max length: {self.max_length}")

    def get_context(self) -> List[dict]: return self.msgs
    def set_context(self, messages: List[dict]): self.msgs = messages
    def clear_context(self, keep_system_prompt: bool = True):
        if keep_system_prompt and self.msgs and self.msgs[0].get("role") == "system":
            self.msgs = [self.msgs[0]]
        else:
            self.msgs = []

    def _count_length(self, messages: List[dict]) -> int:
        if self.use_char_limit:
            return sum(len(str(msg.get("content", ""))) for msg in messages)
        # Using tiktoken for accurate token count
        return sum(len(self.encoding.encode(str(msg.get("content", "")))) for msg in messages)

    def _trim_history(self) -> bool:
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

    async def _clean_response(self, response: str) -> str:
        # Placeholder for ad cleaning logic
        return response

    async def generate(self, msg: Optional[str] = None, *, stream: bool = False, provider: Optional[str] = None, web_search: bool = False, images: Optional[List[Any]] = None, return_format: Literal['tuple', 'content'] = 'tuple', **kwargs) -> Any:

        # 1. Update/Check Model Capabilities
        if provider and provider != self.provider: self._update_model_info(provider)
        if web_search and not self.info.get('supports_web_search'):
            logger.warning(f"Web search not supported by '{self.provider}'. Ignoring request."); web_search = False
        if images and not self.info.get('supports_vision'):
            logger.warning(f"Vision not supported by '{self.provider}'. Ignoring images."); images = None

        max_retries = self.config.get("max_retries")
        timeout = kwargs.get('timeout', self.config.get("timeout"))
        if msg: self.msgs.append({"role": "user", "content": msg})

        # 2. Resilient Request Loop
        g4f_kwargs = {"model": self.model, "messages": self.msgs, "stream": stream, "provider": self.provider, "web_search": web_search, "images": images, **kwargs}
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

                content = ""
                if stream:
                    async for chunk in response:
                        if chunk.choices[0].delta.content: content += chunk.choices[0].delta.content
                else:
                    content = response.choices[0].message.content

                if content:
                    self.msgs.append({"role": "assistant", "content": content})
                    if kwargs.get('clean_ad'):
                        content = await self._clean_response(content)
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