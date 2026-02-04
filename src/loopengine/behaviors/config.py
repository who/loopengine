"""Configuration loading for LLM behavior settings.

This module provides Pydantic-based configuration loading from environment
variables and .env files. API keys are secured by never being logged or
exposed in error messages.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loopengine.behaviors.rate_limiter import RateLimitConfig

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMConfig(BaseSettings):
    """Configuration for LLM provider settings.

    Loads settings from environment variables and .env file.
    API keys are stored as SecretStr to prevent accidental logging.

    Environment Variables:
        LLM_PROVIDER: Which LLM provider to use (claude, openai, ollama)
        ANTHROPIC_API_KEY: Claude API key (required if provider is claude)
        OPENAI_API_KEY: OpenAI API key (required if provider is openai)
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL: Ollama model to use (default: llama2)
        LLM_MAX_TOKENS: Maximum tokens in response (default: 500)
        LLM_TEMPERATURE: Response randomness 0.0-2.0 (default: 0.7)
        LLM_TIMEOUT: Request timeout in seconds (default: 30.0)
        BEHAVIOR_CACHE_TTL: Cache duration in seconds (default: 300)

    Example:
        >>> config = LLMConfig()  # Loads from environment
        >>> config = LLMConfig(_env_file=".env")  # Explicit .env file
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    llm_provider: LLMProvider = Field(
        default=LLMProvider.CLAUDE,
        description="Which LLM provider to use",
    )

    # API Keys (stored as SecretStr for security)
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Claude API key",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key",
    )

    # Ollama settings (for local models)
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="llama2",
        description="Ollama model to use",
    )

    # LLM behavior settings
    llm_max_tokens: int = Field(
        default=500,
        ge=1,
        le=100000,
        description="Maximum tokens in response",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Response randomness (0.0 to 2.0)",
    )
    llm_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds",
    )

    # Caching settings
    behavior_cache_ttl: int = Field(
        default=300,
        ge=0,
        description="Cache duration in seconds (0 to disable)",
    )

    # Rate limit handling settings
    rate_limit_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts on rate limit",
    )
    rate_limit_initial_wait: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial wait time before first retry (seconds)",
    )
    rate_limit_max_wait: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum wait time between retries (seconds)",
    )
    rate_limit_strategy: str = Field(
        default="retry_with_backoff",
        description="Rate limit strategy: retry_with_backoff, fallback_immediately",
    )

    # Concurrency settings for parallel agent decisions (NFR-004)
    max_concurrent_requests: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum concurrent LLM requests (default 50 for NFR-004)",
    )
    request_queue_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Size of request queue for pending behavior requests",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: Any) -> LLMProvider:
        """Normalize provider string to enum."""
        if isinstance(v, str):
            return LLMProvider(v.lower())
        return v

    def get_api_key(self) -> str | None:
        """Get the API key for the current provider.

        Returns:
            The API key string, or None if not configured.
            Never logs the actual key value.
        """
        if self.llm_provider == LLMProvider.CLAUDE:
            return self.anthropic_api_key.get_secret_value() if self.anthropic_api_key else None
        elif self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key.get_secret_value() if self.openai_api_key else None
        else:
            # Ollama doesn't need an API key
            return None

    def validate_config(self) -> None:
        """Validate that required configuration is present.

        Raises:
            ValueError: If required API key is missing for the selected provider.
        """
        if self.llm_provider == LLMProvider.CLAUDE and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'claude'")
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")

    def get_rate_limit_config(self) -> RateLimitConfig:
        """Create a RateLimitConfig from this config's rate limit settings.

        Returns:
            RateLimitConfig instance with current settings.
        """
        from loopengine.behaviors.rate_limiter import RateLimitConfig, RateLimitStrategy

        strategy = RateLimitStrategy(self.rate_limit_strategy)
        return RateLimitConfig(
            strategy=strategy,
            max_retries=self.rate_limit_max_retries,
            initial_wait=self.rate_limit_initial_wait,
            max_wait=self.rate_limit_max_wait,
        )

    def __repr__(self) -> str:
        """Safe representation that never exposes API keys."""
        return (
            f"LLMConfig("
            f"provider={self.llm_provider.value}, "
            f"max_tokens={self.llm_max_tokens}, "
            f"temperature={self.llm_temperature}, "
            f"timeout={self.llm_timeout}s, "
            f"cache_ttl={self.behavior_cache_ttl}s, "
            f"max_concurrent={self.max_concurrent_requests}, "
            f"anthropic_key={'*****' if self.anthropic_api_key else 'not set'}, "
            f"openai_key={'*****' if self.openai_api_key else 'not set'}"
            f")"
        )


@lru_cache
def get_llm_config() -> LLMConfig:
    """Get cached LLM configuration singleton.

    Loads configuration once and caches it for subsequent calls.
    To reload configuration, call get_llm_config.cache_clear() first.

    Returns:
        LLMConfig instance with settings from environment.

    Example:
        >>> config = get_llm_config()
        >>> config.llm_provider
        LLMProvider.CLAUDE
    """
    config = LLMConfig()
    logger.info("Loaded LLM configuration: %s", config)
    return config
