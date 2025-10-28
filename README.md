# G4F-SDK: The Resilient G4F Client

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)

**G4F-SDK** is the missing fault-tolerant SDK for the powerful `g4f` (GPT4Free) library. While `g4f` provides access to a wide range of free AI models, its providers can often be unstable, have undocumented rate limits, and varying context length restrictions.

This module acts as an intelligent wrapper, turning `g4f` into a reliable tool for serious projects by adding layers of resilience, intelligence, and unified API access for Chat, Image Generation, and Audio processing.

## ✨ Key Features

* **🧠 Intelligent Failover & Retries**: Automatically retries requests on timeouts or errors.
* **📏 Adaptive Context Management**: Dynamically detects provider-specific context length limits and reduces the context window to prevent overflow errors.
* **✂️ Smart History Trimming**: Automatically truncates chat history to fit within the model's context window, prioritizing system prompts and recent messages.
* **🔮 Hybrid Model Database**: Combines dynamic discovery of `g4f` models with a rich static database to provide crucial metadata (token limits, vision/web support) even for new providers.
* **🛡️ Feature Awareness**: Prevents errors by checking if a chosen provider supports features like Vision or Web Search before sending the request.
* **📦 Modular & Unified API**: Provides a clean, top-level client (`G4F`) for all functionalities.

## 🔧 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/G4F-SDK.git
    cd G4F-SDK
    ```

2. Create a `requirements.txt` file with the necessary dependencies (or use the one provided):
    ```txt
    # requirements.txt
    g4f
    tiktoken
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start

Here's how easy it is to get a reliable response from a chat model.

```python
import asyncio
# The main client is simply named 'G4F'
from ai import G4F

client = G4F()

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

The main client class is `G4F`.

**1. Default (searches for `config.json`):**
```python
from ai import G4F
client = G4F()
```

**2. With a dictionary:**
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

## 📂 Project Structure

```
.
├── ai/
│ ├── __init__.py  # Main G4F Facade Class
│ ├── config.py  # Config Class and Base Handler
│ ├── chat.py  # ChatHandler (Text and Vision Logic)
│ ├── media.py  # ImageHandler and AudioHandler
│ ├── models_info.py # Hybrid Model Database System
│ └── default_config.py # Fallback default settings
├── .gitignore  # Files to ignore
├── README.md  # You are here!
└── requirements.txt # Project dependencies
```

## ⚙️ Configuration

Create a `config.json` file in your project's root to customize behavior:

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

Contributions are welcome! This project is licensed under the **MIT License**, making it fully open-source and permissible for all types of use.