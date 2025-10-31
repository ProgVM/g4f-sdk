"""
g4f_sdk/config.py

Manages the SDK's configuration with a clear priority system.
"""
import os
import json
import logging
from typing import Optional, Dict, List, Any

class Config:
 """
 Manages all configuration settings for the SDK.
 Priority order:
 1. Keyword arguments during G4FClient initialization (highest).
 2. Settings from a JSON config file.
 3. Default values defined here (lowest).
 """
 def __init__(self, config_path: Optional[str] = None, **kwargs: Any):
  # --- Default Settings ---
  # Logging
  self.log_level: str = "INFO"

  # Core behavior
  self.default_model: str = "gpt-4o"
  self.timeout: int = 120

  # Retry mechanism
  self.retries: int = 3
  self.retry_delay: float = 2.0
  self.retry_backoff_factor: float = 2.0

  # Provider selection
  self.provider_cache_ttl: int = 86400 # 24 hours
  self.preferred_providers: Optional[List[str]] = None

  # Chat specific
  self.use_ai_cleaner: bool = False
  self.max_history_tokens: int = 4096 # Max tokens for chat history before trimming

  # Network
  self.proxy: Optional[Dict[str, str]] = None
  self.api_key: Optional[str] = None

  # --- Load from file ---
  self._load_from_file(config_path)

  # --- Override from kwargs (highest priority) ---
  for key, value in kwargs.items():
   if hasattr(self, key):
    setattr(self, key, value)

 def _load_from_file(self, config_path: Optional[str]):
  """Loads configuration from a .json file if it exists."""
  path_to_check = config_path or "g4f_sdk_config.json"
  if not os.path.exists(path_to_check):
   if config_path: # Warn only if a specific path was given and not found
    logging.warning(f"Configuration file not found: {config_path}")
   return

  try:
   with open(path_to_check, 'r') as f:
    config_data = json.load(f)
   for key, value in config_data.items():
    if hasattr(self, key):
     setattr(self, key, value)
  except (IOError, json.JSONDecodeError) as e:
   logging.error(f"Failed to load or parse config file {path_to_check}: {e}")

 def get(self, key: str, default: Any = None) -> Any:
  """Safely gets a configuration value."""
  return getattr(self, key, default)
