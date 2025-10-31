"""
g4f_sdk/providers.py

Manages, caches, and intelligently selects providers.
"""
import time
import random
import logging
from typing import List, Optional, Dict, Any

from .exceptions import ProviderError

try:
 import g4f
except ImportError:
 g4f = None

logger = logging.getLogger(__name__)

class ProviderManager:
 """
 Manages the list of available providers and selects the best one for a given task.
 """
 def __init__(self, config: Any):
  self.config = config
  self._providers_info: Dict[str, Dict[str, Any]] = {}
  self._last_fetch_time: float = 0

 def _is_cache_valid(self) -> bool:
  """Checks if the provider cache is still valid based on TTL."""
  return time.time() - self._last_fetch_time < self.config.provider_cache_ttl

 def _fetch_from_g4f(self):
  """
  Fetches and updates the list of providers by introspecting the g4f library.
  """
  if not g4f:
   return

  logger.info("Refreshing provider cache from g4f library...")
  self._providers_info.clear()

  for provider_name in dir(g4f.Provider):
   if not provider_name.startswith("__"):
    provider_class = getattr(g4f.Provider, provider_name)
    if isinstance(provider_class, type) and getattr(provider_class, "working", False):
     self._providers_info[provider_name] = {
      "name": provider_name,
      "working": True,
      "models": getattr(provider_class, "model", None)
     }

  self._last_fetch_time = time.time()
  logger.info(f"Cache updated. Found {len(self._providers_info)} working providers.")

 def _ensure_cache(self):
  """Ensures the provider cache is fresh before use."""
  if not self._is_cache_valid():
   self._fetch_from_g4f()

 def _select_provider(self, model: str, provider_hint: Optional[str] = None) -> str:
  """
  Selects the best available provider for a given model.

  Selection logic:
  1. If a specific provider is hinted, use it if it's working.
  2. Check preferred providers from config that explicitly support the model.
  3. Check any working provider that explicitly supports the model.
  4. As a fallback, check any working preferred provider (even without explicit model support).
  5. As a final fallback, choose a random working provider.
  """
  self._ensure_cache()

  if provider_hint:
   if provider_hint in self._providers_info:
    return provider_hint
   else:
    raise ProviderError(f"Specified provider '{provider_hint}' is not available or not working.")

  working_providers = list(self._providers_info.keys())
  if not working_providers:
   raise ProviderError("No working providers found in the cache.")

  def check_model_support(provider_name: str) -> bool:
   """Checks if a provider explicitly supports the model."""
   provider_models = self._providers_info[provider_name].get("models")
   if isinstance(provider_models, str):
    return provider_models == model
   if isinstance(provider_models, list):
    return model in provider_models
   return False

  # 1. & 2. Find candidates from preferred list first
  preferred = self.config.preferred_providers or []
  candidates = [p for p in preferred if p in working_providers and check_model_support(p)]

  # 3. Find candidates from all working providers
  if not candidates:
   candidates = [p for p in working_providers if check_model_support(p)]

  # 4. Fallback to any preferred provider
  if not candidates and preferred:
   candidates = [p for p in preferred if p in working_providers]

  # 5. Fallback to any working provider
  if not candidates:
   candidates = working_providers

  return random.choice(candidates)