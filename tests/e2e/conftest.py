"""Pytest configuration for e2e tests.

This module provides shared fixtures and markers for e2e tests.
"""

from __future__ import annotations

import os

import pytest

# Marker for tests that require an Anthropic API key
requires_api_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping AI-dependent test",
)
"""Skip marker for tests that require LLM API access.

Apply this decorator to any test that:
- Calls /api/discovery/run and expects successful completion
- Calls any endpoint that uses LLM-based behavior generation
- Depends on AIPolicy or AIBehaviorEngine functionality

Example usage:
    from tests.e2e.conftest import requires_api_key

    @requires_api_key
    def test_discovery_completes_successfully():
        # This test will be skipped if ANTHROPIC_API_KEY is not set
        ...

Note: Tests that only verify endpoint existence (non-404) or
validation behavior (400 for invalid input) do NOT need this marker,
as they don't require actual LLM calls.
"""
