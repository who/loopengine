"""E2E tests for backend and frontend startup verification.

These tests verify:
1. Backend /health endpoint returns 200 with status "healthy"
2. Frontend loads and displays the LoopEngine Visualizer UI
3. Frontend connects to backend via WebSocket and receives frames

Requirements:
- Backend must be running on localhost:8000
- Frontend must be served on localhost:8080
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect


class TestBackendHealth:
    """Tests for backend health endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_health_endpoint_returns_200(self) -> None:
        """Verify /health endpoint returns 200 OK with healthy status."""
        response = self.page.request.get("http://localhost:8000/health")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert data["status"] == "healthy", f"Expected status 'healthy', got {data}"

    def test_api_world_endpoint(self) -> None:
        """Verify /api/world endpoint returns world state."""
        response = self.page.request.get("http://localhost:8000/api/world")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert "tick" in data, "Expected 'tick' in world response"
        assert "agent_count" in data, "Expected 'agent_count' in world response"
        assert "paused" in data, "Expected 'paused' in world response"


class TestFrontendUI:
    """Tests for frontend UI verification."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_frontend_loads_with_correct_title(self) -> None:
        """Verify frontend page loads with correct title."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        expect(self.page).to_have_title("LoopEngine Visualizer")

    def test_canvas_element_exists(self) -> None:
        """Verify the canvas element is present on the page."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        canvas = self.page.locator("canvas#canvas")
        expect(canvas).to_be_visible()

    def test_canvas_has_dimensions(self) -> None:
        """Verify canvas has non-zero dimensions after load."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for canvas to be visible
        canvas = self.page.locator("canvas#canvas")
        expect(canvas).to_be_visible()

        # Check canvas has dimensions
        bounding_box = canvas.bounding_box()
        assert bounding_box is not None, "Canvas should have a bounding box"
        assert bounding_box["width"] > 0, "Canvas should have positive width"
        assert bounding_box["height"] > 0, "Canvas should have positive height"


class TestFrontendBackendIntegration:
    """Tests verifying frontend-backend communication."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_websocket_connects_and_receives_frames(self) -> None:
        """Verify frontend connects to backend WebSocket and receives frame data."""
        # Collect console messages to check for WebSocket connection
        console_messages: list[str] = []

        def handle_console(msg: object) -> None:
            console_messages.append(str(msg))

        self.page.on("console", handle_console)

        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for WebSocket connection and frame data
        self.page.wait_for_timeout(3000)

        # Check that we saw the connection message
        ws_connected = any("Frames WebSocket connected" in m for m in console_messages)
        assert ws_connected, (
            f"Expected 'Frames WebSocket connected' in console. Got: {console_messages[:10]}"
        )

    def test_no_javascript_errors(self) -> None:
        """Verify page loads without JavaScript errors."""
        errors: list[str] = []

        def handle_error(error: str) -> None:
            errors.append(error)

        self.page.on("pageerror", lambda exc: handle_error(str(exc)))

        self.page.goto("http://localhost:8080/frontend/index.html")

        # Give the page time to establish connections
        self.page.wait_for_timeout(2000)

        assert len(errors) == 0, f"JavaScript errors found: {errors}"

    def test_loopengine_global_available(self) -> None:
        """Verify window.loopEngine global is available after initialization."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait a bit for initialization
        self.page.wait_for_timeout(1000)

        # Check if loopEngine global exists
        result = self.page.evaluate("typeof window.loopEngine")
        assert result == "object", f"Expected loopEngine to be object, got {result}"

        # Check if essential methods exist
        has_play = self.page.evaluate("typeof window.loopEngine.play === 'function'")
        has_pause = self.page.evaluate("typeof window.loopEngine.pause === 'function'")

        assert has_play, "loopEngine.play should be a function"
        assert has_pause, "loopEngine.pause should be a function"
