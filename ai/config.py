# ai/config.py

import logging
import json
import importlib.util
from typing import Optional, Union, Any
from pathlib import Path

from g4f.client import Client, AsyncClient
from . import default_config

logger = logging.getLogger("g4f-sdk.config")

# --- Smart Configuration Class ---
class Config:
    """Handles loading and accessing configuration from various sources."""
    def __init__(self, config_input: Optional[Union[str, dict, object]] = None, **kwargs):
        self.data = self._load_defaults()
        user_config = self._load_from_input(config_input)

        if user_config:
            self.data.update(user_config)

        if kwargs:
            self.data.update(kwargs)

    def _load_defaults(self) -> dict:
        return {
            attr: getattr(default_config, attr)
            for attr in dir(default_config) if not attr.startswith('__')
        }

    def _load_from_input(self, config_input: Any) -> Optional[dict]:
        if config_input is None:
            config_path = Path("config.json")
            if config_path.exists():
                logger.info("Loading configuration from config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        if isinstance(config_input, dict):
            return config_input

        if isinstance(config_input, str):
            path = Path(config_input)
            if not path.exists():
                logger.error(f"Config file not found at: {config_input}")
                return None
            if path.suffix == '.json':
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            if path.suffix == '.py':
                spec = importlib.util.spec_from_file_location("custom_config", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return {attr: getattr(module, attr) for attr in dir(module) if not attr.startswith('__')}

        if hasattr(config_input, '__dict__'):
            return {attr: getattr(config_input, attr) for attr in dir(config_input) if not attr.startswith('__')}

        logger.warning(f"Unsupported config input type: {type(config_input)}")
        return None

    def get(self, key: str, default=None):
        return self.data.get(key, default)

# --- Base Handler with common logic ---
class _BaseHandler:
    """Provides common properties and clients for all specific handlers."""
    def __init__(self, config: Config):
        self.config = config
        self.api_key = self.config.get("api_key")
        self.client = Client(api_key=self.api_key)
        self.async_client = AsyncClient(api_key=self.api_key)