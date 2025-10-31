"""
g4f_sdk/exceptions.py

Defines detailed, custom exceptions for the SDK for better error handling.
"""
from typing import Optional

class G4FSDKError(Exception):
 """Base exception for all SDK errors."""
 pass

class ConfigurationError(G4FSDKError):
 """Exception raised for configuration-related errors."""
 pass

class ProviderError(G4FSDKError):
 """
 Exception raised for errors related to a specific provider during an operation.
 This is a broad category for when a provider fails for a specific reason.
 """
 def __init__(self, message: str, provider_name: Optional[str] = None):
  self.provider_name = provider_name
  full_message = f"Provider '{provider_name}': {message}" if provider_name else message
  super().__init__(full_message)

class RateLimitError(ProviderError):
 """
 Exception raised when a rate limit is exceeded for a provider.
 This is a retryable error.
 """
 def __init__(self, message: str, provider_name: Optional[str] = None):
  super().__init__(message, provider_name)

class InvalidResponseError(ProviderError):
 """
 Exception raised for invalid, empty, or unexpected responses from a provider.
 This might be retryable, as it could be a temporary issue.
 """
 def __init__(self, message: str, provider_name: Optional[str] = None):
  super().__init__(message, provider_name)

class ModelNotFoundError(G4FSDKError):
 """Exception raised when a requested model cannot be found or is not supported."""
 def __init__(self, model_name: str):
  self.model_name = model_name
  super().__init__(f"Model not found or not supported by any available provider: '{model_name}'")

class APIError(G4FSDKError):
 """
 A generic exception for when an API call fails after all retries.
 This wraps the last error encountered."""
 def __init__(self, message: str, last_exception: Optional[Exception] = None):
  self.last_exception = last_exception
  super().__init__(message)
