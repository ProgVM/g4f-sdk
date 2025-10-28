# ai/models_info.py

import logging
from typing import Optional, List, Dict, Any, Union
from g4f import models
from g4f.Provider import __all__ as all_providers

logger = logging.getLogger("g4f-sdk.models_info")

# STEP 1: STATIC ENRICHMENT DATABASE
# This is our "knowledge layer". We store information here that g4f doesn't provide directly.
# Structure: 'model' -> '_default' (settings for all model providers)
#   -> 'providers' -> 'provider' (provider-specific settings)
STATIC_ENRICHMENT_DATA = {
    'gpt-4o': {
        '_default': {'type': 'vision', 'max_tokens': 128000, 'supports_vision': True},
        'providers': {
            'Bing': {'max_tokens': 32768, 'supports_web_search': True, 'is_stable': True, 'notes': 'Best all-around provider.'},
            'GptGo': {'max_tokens': 8192, 'supports_web_search': True, 'is_stable': True},
            'You': {'max_tokens': 4096, 'supports_web_search': True, 'is_stable': False, 'notes': 'Often unstable.'},
        }
    },
    'claude-3-opus': {
        '_default': {'type': 'vision', 'max_tokens': 200000, 'supports_vision': True},
        'providers': {
            'ClaudeDev': {'is_stable': True, 'notes': 'Official Anthropic provider.'},
            'Poe': {'max_tokens': 8192, 'is_stable': False, 'notes': 'Requires cookies.'},
        }
    },
    'gemini': {
        '_default': {'type': 'vision', 'max_tokens': 32768, 'supports_vision': True},
        'providers': {
            'Google': {'is_stable': True, 'notes': 'Official Google provider.'},
            'GeminiAdvanced': {'max_tokens': 128000, 'is_stable': False, 'notes': 'Requires cookies.'},
        }
    },
    'dall-e-3': {
        '_default': {'type': 'image'},
        'providers': { 'Bing': {'is_stable': True}, 'GptGo': {'is_stable': True} }
    },
    'whisper-1': { '_default': {'type': 'audio_transcription'} },
    'tts-1': { '_default': {'type': 'audio_speech', 'max_chars': 4096} }
}

# GENERIC DEFAULTS for models or providers not found in the static enrichment data.
GENERIC_DEFAULTS = {
    'type': 'chat',
    'max_tokens': 99999,
    'supports_vision': True,
    'supports_web_search': True,
    'is_stable': False,
    'notes': 'No static info available for this model/provider.'
}

_models_info_cache = None

def _discover_and_merge_models_info():
    """Scans g4f and merges the findings with our static knowledge layer."""
    logger.info("Discovering g4f models and enriching with static data...")
    merged_models = {}

    # Dynamically get all models from g4f
    for attr_name in dir(models):
        if attr_name.startswith('__'): continue
        model_obj = getattr(models, attr_name)
        if hasattr(model_obj, 'name') and hasattr(model_obj, 'best_provider'):
            model_name = model_obj.name

            # Get a list of all providers for this model
            provider_names = [p.__name__ for p in getattr(model_obj, 'providers', [])]
            if not provider_names and hasattr(model_obj, 'best_provider'):
                provider_names = [model_obj.best_provider.__name__]

            merged_models[model_name] = {
                'name': model_name,
                'g4f_best_provider': model_obj.best_provider.__name__,
                'providers': {}
            }

            static_model_data = STATIC_ENRICHMENT_DATA.get(model_name, {})
            model_defaults = static_model_data.get('_default', {})
            static_providers_data = static_model_data.get('providers', {})

            # For each provider, create a complete profile
            for provider_name in provider_names:
                # Build the profile in layers: Generic -> Model Default -> Provider Specific
                provider_profile = GENERIC_DEFAULTS.copy()
                provider_profile.update(model_defaults)
                provider_profile.update(static_providers_data.get(provider_name, {}))

                merged_models[model_name]['providers'][provider_name] = provider_profile

    logger.info(f"Discovered and processed {len(merged_models)} models.")
    return merged_models

def get_all_models_info() -> dict:
    """Returns a complete, cached dictionary with all model information."""
    global _models_info_cache
    if _models_info_cache is None:
        _models_info_cache = _discover_and_merge_models_info()
    return _models_info_cache

def get_model_provider_info(model_name: str, provider_name: Optional[str] = None) -> dict:
    """
    Returns detailed information for a specific model-provider pair.

    If a provider is explicitly named, this function ensures that provider is used,
    even if it is not in the static configuration for the model, to prevent
    unwanted fallback to RetryProvider.
    """
    all_models = get_all_models_info()
    model_info = all_models.get(model_name)

    # 1. Handle case where model is not found in g4f dynamic list
    if not model_info:
        logger.warning(f"Model '{model_name}' not found in g4f. Using generic defaults.")
        chosen_provider = provider_name or None 
        return {'model_name': model_name, 'chosen_provider': chosen_provider, **GENERIC_DEFAULTS}

    # 2. Determine the chosen provider
    chosen_provider = provider_name

    # If a provider name was explicitly passed
    if provider_name:
        # Check if the explicitly requested provider is in our static/dynamic info for this model
        if provider_name not in model_info['providers']:
            # If the provider is explicitly requested but not in our known list for this model,
            # we check if it is a known g4f provider (by checking provider_name in __all__).
            # Note: all_providers contains classes, we check against their __name__.
             is_valid_g4f_provider_name = any(provider_name == p.__name__ for p in all_providers)

             if is_valid_g4f_provider_name:
                 logger.warning(f"Provider '{provider_name}' explicitly requested but no static info available for model '{model_name}'. Forcing its use with generic defaults.")
                 # We return immediately with generic defaults to force use of the requested provider
                 return {
                     'model_name': model_name,
                     'chosen_provider': provider_name,
                     **GENERIC_DEFAULTS
                 }
             else:
                 # If the explicitly requested provider is not even a valid g4f provider, fall back.
                 logger.warning(f"Explicitly requested provider '{provider_name}' is not a valid g4f provider. Falling back to g4f best provider.")
                 chosen_provider = model_info['g4f_best_provider']

    # If no provider was specified, or if we fell back from an invalid one
    if not chosen_provider:
        # If no provider is specified, try to find a stable default one or g4f's best.
        static_model_data = STATIC_ENRICHMENT_DATA.get(model_name, {})
        chosen_provider = static_model_data.get('_default', {}).get('default_provider') or model_info['g4f_best_provider']

    # 3. Get the final provider information
    provider_info = model_info['providers'].get(chosen_provider)

    # Fallback in case chosen_provider is still not found in the dictionary keys
    if not provider_info:
        logger.warning(f"Provider '{chosen_provider}' is still not found for model '{model_name}'. Using generic defaults.")
        provider_info = GENERIC_DEFAULTS.copy()

    return {
        'model_name': model_name,
        'chosen_provider': chosen_provider,
        **provider_info
    }
