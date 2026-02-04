"""AI genome discovery via LLM-based schema inference."""

from loopengine.discovery.discoverer import (
    VALID_CATEGORIES,
    DiscoveredRole,
    Discoverer,
    DiscoveryError,
    DiscoveryResult,
    MigrationResult,
    discover_schemas,
    migrate_genome,
)
from loopengine.discovery.triggers import (
    RediscoveryTriggerManager,
    ScheduledTriggerConfig,
    StagnationConfig,
    TriggerEvent,
    TriggerType,
    extract_system_description,
)

__all__ = [
    "VALID_CATEGORIES",
    "DiscoveredRole",
    "Discoverer",
    "DiscoveryError",
    "DiscoveryResult",
    "MigrationResult",
    "RediscoveryTriggerManager",
    "ScheduledTriggerConfig",
    "StagnationConfig",
    "TriggerEvent",
    "TriggerType",
    "discover_schemas",
    "extract_system_description",
    "migrate_genome",
]
