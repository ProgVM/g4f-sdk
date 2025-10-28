# ai/__init__.py

from typing import Optional, List, Union, Any
import logging

from .config import Config
from .chat import ChatHandler
from .media import ImageHandler, AudioHandler
from g4f.models import gpt_4o

logging.getLogger("g4f-sdk").setLevel(logging.INFO)

# --- Main Client Facade Class ---
class G4F:
    """
    The main client for the G4F-SDK. Provides resilient access to 
    chat completions, image generation, and audio processing.
    """
    def __init__(self, config_input: Optional[Union[str, dict, object]] = None):
        """
        Initializes the G4F-SDK client.
        :param config_input: Can be a dictionary, a path to a .json/.py file, or an object.
        """
        self.config = Config(config_input)
        self.chat = ChatHandler(self.config)
        self.images = ImageHandler(self.config)
        self.audio = AudioHandler(self.config)

    def new_chat(self, model=gpt_4o, context: Optional[List[dict]] = None) -> ChatHandler:
        """
        Creates a new, independent chat instance.
        :param model: The g4f model object to use for the chat.
        :param context: An optional list of messages to initialize the chat history.
        :return: A new instance of ChatHandler.
        """
        return ChatHandler(self.config, model=model, context=context)