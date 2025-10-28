# G4F-SDK: The Resilient G4F Client

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyPI Version](https://img.shields.io/pypi/v/g4f-sdk)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)

**G4F-SDK** is the missing fault-tolerant SDK for the powerful `g4f` (GPT4Free) library. While `g4f` provides access to a wide range of free AI models, its providers can often be unstable, have undocumented rate limits, and varying context length restrictions.

This module acts as an intelligent wrapper, turning `g4f` into a reliable tool for serious projects by adding layers of resilience, intelligence, and unified API access for Chat, Image Generation, and Audio processing.

## ✨ Key Features

* **🧠 Intelligent Failover & Retries**: Automatically retries requests on timeouts or errors.
* **📏 Adaptive Context Management**: Dynamically detects provider-specific context length limits and reduces the context window to prevent overflow errors.
* **✂️ Smart History Trimming**: Automatically truncates chat history to fit within the model's context window, prioritizing system prompts and recent messages.
* **🔮 Hybrid Model Database**: Combines dynamic discovery of `g4f` models with a rich static database to provide crucial metadata (token limits, vision/web support) even for new providers.
* **🛡️ Feature Awareness**: Prevents errors by checking if a chosen provider supports features like Vision or Web Search before sending the request.
* **📦 Modular & Unified API**: Provides a clean, top-level client (`G4F`) for all functionalities with flexible configuration options.

## 🔧 Installation

The G4F-SDK is available on PyPI.

1. Install the package using pip:
    ```bash
    pip install g4f-sdk
    ```
    *(Note: This command automatically installs the core dependencies: `g4f` and `tiktoken`.)*

2. *(Optional)* Create a `config.json` file in your project root to customize settings.

## 🚀 Quick Start

Initialize the client and run a resilient chat completion.

```python
import asyncio
# Import the main client class G4F from the installed package
from ai import G4F

# Configuration can be passed directly via kwargs
client = G4F(timeout=60, max_retries=3)

async def main():
    print("--- Starting a simple chat ---")

 # The generate method is resilient to failures
    response_content, updated_context = await client.chat.generate(
        msg="Hello! Can you tell me a fun fact about programming?"
    )

    if response_content:
        print("\nAI Response:")
        print(response_content)
    else:
        print("\nFailed to get a response after several retries.")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📚 Full Usage Guide

### Initializing the Client

The main client class is `G4F`. Configuration can be passed in three ways (in order of priority: kwargs > config_input > config.json).

**1. Default (searches for `config.json`):**
```python
from ai import G4F
client = G4F()
```

**2. With direct Keyword Arguments (kwargs):**
(Recommended for quick settings override)
```python
client = G4F(timeout=60, max_retries=3)
```

**3. With a dictionary (config_input):**
(Useful for dynamic configuration)
```python
custom_config = {"timeout": 60, "max_retries": 3}
client = G4F(config_input=custom_config)
```

### Chat Completions

The `client.chat` object handles all text and vision tasks.

**Simple Text Generation:**
```python
response, context = await client.chat.generate(msg="What is the capital of France?")
print(response)
```

**Resilient Streaming Completions**
For real-time output, use `client.chat.stream_generate`. This dedicated asynchronous generator method is fully **fault-tolerant**: if a provider fails mid-stream, the SDK will automatically retry the generation on a new provider, seamlessly attempting to continue the text from where it broke off.

```python
import asyncio
from g4f_sdk.ai import G4F

async def stream_example():
    client = G4F(max_retries=5)

    print("Streaming response (Resilient mode):")

    # The async generator handles retries internally if the connection drops
    async for chunk in client.chat.stream_generate(
        msg="Write a short, detailed story about an AI who discovered music."
    ):
        print(chunk, end="", flush=True)

    print("\n--- Stream finished ---")

if __name__ == "__main__":
    asyncio.run(stream_example())
 

**Using Vision (Image Input):**
```python
from g4f.models import gpt_4o
import base64

# Use client.new_chat() for a specific model/context
vision_chat = client.new_chat(model=gpt_4o)

with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

response, _ = await vision_chat.generate(
    msg="What is in this image?",
    images=[f"data:image/jpeg;base64,{image_base64}"]
)
print(response)
```

**Using Web Search (RAG):**
```python
response, _ = await client.chat.generate(
    msg="What are the latest news on AI?",
    web_search=True
)
print(response)
```

### Image Generation

Use the `client.images` object.

```python
image_url = await client.images.generate(
    prompt="A cute robot programming on a laptop, digital art",
    nologo=True 
)
if image_url:
    print(f"Image generated: {image_url}")
```

### Audio Processing

Use the `client.audio` object for Text-to-Speech and Speech-to-Text.

**Text-to-Speech (TTS):**
```python
audio_bytes = await client.audio.text_to_speech(
    text="Hello world! This is a test of the text-to-speech system."
)
if audio_bytes:
    with open("output.mp3", "wb") as f:
        f.write(audio_bytes)
    print("Saved speech to output.mp3")
```

**Speech-to-Text (STT):**
```python
transcribed_text = await client.audio.speech_to_text(file="output.mp3")
if transcribed_text:
    print(f"Transcribed text: '{transcribed_text}'")
```

### Managing Chat Context

You can create multiple independent chat sessions and manage their history.

```python
# Create two separate conversations
chat_1 = client.new_chat()
chat_2 = client.new_chat()

await chat_1.generate(msg="My name is Bob.")
await chat_2.generate(msg="My name is Alice.")

# Ask chat 1 about its context
response, _ = await chat_1.generate(msg="What is my name?")
print(f"Chat 1 response: {response}")

# Ask chat 2 about its context
response, _ = await chat_2.generate(msg="What is my name?")
print(f"Chat 2 response: {response}")

# You can also manually get or set the context
current_history = chat_1.get_context()
print(current_history)
```

## 📂 Project Structure

```
.
├── ai/
│ ├── __init__.py # Main G4F Facade Class (imported as 'from ai import G4F')
│ ├── config.py # Config Class and Base Handler
│ ├── chat.py # ChatHandler (Text and Vision Logic)
│ ├── media.py # ImageHandler and AudioHandler
│ ├── models_info.py # Hybrid Model Database System
│ └── default_config.py # Fallback default settings
├── .gitignore # Files to ignore
├── README.md # You are here!
├── requirements.txt # Project dependencies
├── setup.py # Legacy build file
└── pyproject.toml # Modern build configuration (PEP 518/621)
```

## ⚙️ Configuration

You can override settings via `kwargs` at initialization or create a `config.json` file in your project's root:

**Example `config.json`:**
```json
{
    "api_key": null,
    "max_retries": 3,
    "timeout": 60,
    "context_reduction_factor": 0.5
}
```

## 🤝 Contributing & License

Contributions are welcome! This project is licensed under the **MIT License**.