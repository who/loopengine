"""AI behavior system for dynamic LLM-driven agent behaviors."""

from loopengine.behaviors.ai_behavior_engine import (
    AIBehaviorEngine,
    AIBehaviorEngineError,
)
from loopengine.behaviors.config import (
    LLMConfig,
    LLMProvider,
    get_llm_config,
)
from loopengine.behaviors.fallback import (
    FallbackBehavior,
    FallbackReason,
    classify_error,
)
from loopengine.behaviors.llm_client import (
    BehaviorResponse,
    LLMClient,
    LLMClientSettings,
    LLMQuery,
)
from loopengine.behaviors.prompt_builder import (
    AgentContext,
    DomainContext,
    PromptBuilder,
)
from loopengine.behaviors.response_parser import (
    ResponseParser,
    ResponseParserError,
)

__all__ = [
    "AIBehaviorEngine",
    "AIBehaviorEngineError",
    "AgentContext",
    "BehaviorResponse",
    "DomainContext",
    "FallbackBehavior",
    "FallbackReason",
    "LLMClient",
    "LLMClientSettings",
    "LLMConfig",
    "LLMProvider",
    "LLMQuery",
    "PromptBuilder",
    "ResponseParser",
    "ResponseParserError",
    "classify_error",
    "get_llm_config",
]
