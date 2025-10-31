"""
g4f-sdk

A Python SDK for the g4f library with enhanced resilience and configuration.
"""

__version__ = "0.2.0" # Version bump to reflect major improvements

from .client import G4FClient
from .exceptions import (
 G4FSDKError,
 ConfigurationError,
 ProviderError,
 RateLimitError,
 InvalidResponseError,
 ModelNotFoundError,
 APIError
)

__all__ = [
 "G4FClient",
 "G4FSDKError",
 "ConfigurationError",
 "ProviderError",
 "RateLimitError",
 "InvalidResponseError",
 "ModelNotFoundError",
 "APIError",
]
