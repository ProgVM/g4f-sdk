# ai/default_config.py

# --- Default Configuration for G4F-SDK ---

# API Key (for external services, e.g., OpenAI or Firecrawl)
api_key = None

# Default model if not specified in G4F init (must be a string matching a g4f model name)
default_model = "gemini-2.5-flash"

# Default timeout for most API calls (in seconds)
timeout = 60

# Maximum number of retries for resilient API calls
max_retries = 3

# Context reduction factor for history trimming:
# - If >= 1.0, it's the number of messages to keep (e.g., 10 means keep the last 10 messages).
# - If < 1.0, it's the target percentage of max_length to trim context to (e.g., 0.8 means trim to 80% of max tokens).
context_reduction_factor = 0.7

# Default configuration for media generation (e.g., image)
image_generation_model = "flux"
image_generation_size = "1024x1024"
image_generation_quality = "standard"