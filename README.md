# G4F SDK

A resilient and configurable Python SDK for the `g4f` library.

This SDK acts as a robust wrapper around the powerful `g4f` library, providing a simplified interface with built-in resilience, intelligent provider selection, and enhanced configuration management.

## Key Features

- **Smart Provider Selection**: Automatically chooses a working provider, prioritizing those that explicitly support the requested model.
- **Robust Error Handling**: Automatically retries requests on temporary errors (like rate limits or network issues) with configurable exponential backoff.
- **Detailed Exceptions**: Provides specific exceptions like `RateLimitError` and `InvalidResponseError` for fine-grained error handling in your application.
- **Automatic History Trimming**: For chat sessions, the history is automatically trimmed based on token count to prevent exceeding model context limits. Uses `tiktoken` if installed for best accuracy.
- **Highly Configurable**: Control everything from timeouts and retry attempts to preferred providers and chat history length via a single config file or direct arguments.
- **Full API Support**: Unified access to chat, image generation, and audio transcription/synthesis.

## Installation

The SDK requires Python 3.8 or higher.

```bash
pip install g4f-sdk
```

For the most accurate and efficient chat history management, it is highly recommended to install `tiktoken`:

```bash
pip install tiktoken
```

## Quick Start

```python
import asyncio
from g4f_sdk import G4FClient, RateLimitError, APIError

async def main():
 # Initialize the client.
 # It will automatically use defaults or load from `g4f_sdk_config.json` if it exists.
 client = G4FClient(log_level="DEBUG")

 # --- Chat Example ---
 # The ChatSession automatically handles history trimming.
 try:
  print("Starting chat session...")
  chat_session = client.new_chat(system_prompt="You are a helpful assistant.")
  response = await chat_session.generate("Hello, what is the capital of France?")
  print(f"AI Response: {response}")

  # Continue the conversation
  response_2 = await chat_session.generate("And what is its population?")
  print(f"AI Response: {response_2}")

 except RateLimitError as e:
  print(f"Chat failed due to rate limiting by provider '{e.provider_name}'.")
 except APIError as e:
  print(f"Chat failed after all retries. Last error: {e.last_exception}")
 except Exception as e:
  print(f"An unexpected error occurred: {e}")

 # --- Image Generation Example ---
 try:
  print("\nGenerating image...")
  image_url = await client.generate_image("A cute robot programming on a laptop, digital art")
  print(f"Generated Image URL: {image_url}")
 except APIError as e:
  print(f"Image generation failed: {e}")

if __name__ == "__main__":
 asyncio.run(main())
```

## Configuration

You can configure the client in three ways, with the following priority:

1. **Keyword Arguments (Highest Priority)**:
  ```python
  client = G4FClient(timeout=60, retries=5, default_model="gpt-3.5-turbo")
  ```

2. **`g4f_sdk_config.json` file**: Create this file in your project's root directory. The client will automatically find and load it.

  **Example `g4f_sdk_config.json`:**
  ```json
  {
   "log_level": "INFO",
   "default_model": "gpt-4o",
   "timeout": 120,
   "retries": 3,
   "retry_delay": 2.0,
   "retry_backoff_factor": 2.0,
   "provider_cache_ttl": 86400,
   "preferred_providers": [
    "FreeGpt",
    "You"
   ],
   "use_ai_cleaner": false,
   "max_history_tokens": 4096,
   "proxy": null
  }
  ```

3. **Default Settings (Lowest Priority)**: If no other configuration is provided, the SDK uses built-in defaults.

### All Configuration Options

-  `log_level` (str): Logging level (e.g., "DEBUG", "INFO", "WARNING"). Default: "INFO".
-  `default_model` (str): Default model for chat and other tasks. Default: "gpt-4o".
-  `timeout` (int): Request timeout in seconds. Default: 120.
-  `retries` (int): Number of retries on failure. Default: 3.
-  `retry_delay` (float): Initial delay between retries in seconds. Default: 2.0.
-  `retry_backoff_factor` (float): Multiplier for delay on subsequent retries (e.g., 2.0 for exponential backoff). Default: 2.0.
-  `provider_cache_ttl` (int): Time in seconds to cache the list of working providers. Default: 86400 (24 hours).
-  `preferred_providers` (list[str]): A list of provider names to try first.
-  `use_ai_cleaner` (bool): Whether to use an AI call to clean responses from provider artifacts. Default: `false`.
-  `max_history_tokens` (int): The maximum number of tokens to keep in a chat session's history. Default: 4096.
-  `proxy` (dict): Proxy settings (e.g., `{"http": "...", "https": "..."}`).
-  `api_key` (str): A global API key if required by certain providers.

## Advanced Usage: Error Handling

The SDK provides detailed exceptions, allowing you to build robust applications.

```python
from g4f_sdk import G4FClient, RateLimitError, InvalidResponseError, ProviderError, APIError

client = G4FClient()

try:
  response = await client.new_chat().generate("Some complex prompt...")
except RateLimitError as e:
  # Specific action for when you are rate-limited
  print(f"Provider '{e.provider_name}' rate limited us. Waiting before next request.")
except InvalidResponseError as e:
  # The provider returned garbage or an empty response
  print(f"Provider '{e.provider_name}' gave an invalid response.")
except ProviderError as e:
  # A generic error from the provider
  print(f"Provider '{e.provider_name}' failed: {e}")
except APIError as e:
  # This is the final error after all retries have failed
  print(f"The API call failed completely. The last error was: {e.last_exception}")
```

## License

This project is licensed under the MIT License.