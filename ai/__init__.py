# ai/__init__.py

from typing import Optional, List, Union, Any
import logging

from .config import Config
from .chat import ChatHandler
from .media import ImageHandler, AudioHandler
from g4f.models import gpt_4o, gemini_2_5_flash, Model

# Set up logging for the SDK
logging.getLogger("g4f-sdk").setLevel(logging.INFO)

def _get_g4f_provider_class(name: str) -> Optional[BaseProvider]:
 """Dynamically converts a string name into a g4f Provider class."""
 from g4f import Provider # Local import

 for p_class in all_providers:
  if p_class.__name__ == name:
   return p_class
 return None

def _get_g4f_model_reference(name: str) -> Union[Model, str, None]:
  """
  Dynamically gets the g4f model object or returns the string name.

  Tries to find the official g4f.models object first. If not found,
  returns the string name, which g4f.client can often handle directly.
  """
  from g4f import models # Local import

  # 1. Try to find the official g4f Model object (e.g., gpt_4o)
  for attr_name in dir(models):
    attr = getattr(models, attr_name)
    if hasattr(attr, 'name') and attr.name == name:
      return attr

  # 2. Fallback: Return the string name.
  return name

# --- Main Client Facade Class ---
class G4F:
  """
  The main client for the G4F-SDK. Provides resilient access to
  chat completions, image generation, and audio processing.
  """
  def __init__(self, config_input: Optional[Union[str, dict, object]] = None, **kwargs):

    # 1. Extract and process model from kwargs
    model_ref = kwargs.pop('model', None)

    self.config = Config(config_input, **kwargs)

    # 2. Determine the final model reference (object or string)
    if not model_ref:
      model_ref = self.config.get("default_model", "gemini-2.5-flash")

    model_ref = _get_g4f_model_reference(model_ref)

    if isinstance(model_ref, str):
       # Final check: if it's a string and we can't find the model, issue a warning.
       # We still use the string, trusting g4f.client.
       logging.info(f"Using model name string '{model_ref}'.")

    self.chat = ChatHandler(self.config, model=model_ref)
    self.images = ImageHandler(self.config)
    self.audio = AudioHandler(self.config)

  def new_chat(self, model: Union[str, Model] = gemini_2_5_flash, context: Optional[List[dict]] = None) -> ChatHandler:
    """
    Creates a new, independent chat instance.

    :param model: The g4f model object or name (str) to use for the chat.
    :param context: An optional list of messages to initialize the chat history.
    :return: A new instance of ChatHandler.
    """
    model_ref = model
    if isinstance(model, str):
      model_ref = _get_g4f_model_reference(model)

    return ChatHandler(self.config, model=model_ref, context=context)