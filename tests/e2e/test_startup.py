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


class TestAPIEndpoints:
    """Tests verifying all backend API endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_api_agents_endpoint(self) -> None:
        """Verify /api/agents endpoint returns list of agents."""
        response = self.page.request.get("http://localhost:8000/api/agents")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert isinstance(data, list), "Expected list of agents"
        assert len(data) > 0, "Expected at least one agent"

        # Check agent structure
        agent = data[0]
        assert "id" in agent, "Expected 'id' in agent"
        assert "name" in agent, "Expected 'name' in agent"
        assert "role" in agent, "Expected 'role' in agent"

    def test_api_links_endpoint(self) -> None:
        """Verify /api/links endpoint returns list of links."""
        response = self.page.request.get("http://localhost:8000/api/links")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert isinstance(data, list), "Expected list of links"

    def test_api_schemas_endpoint(self) -> None:
        """Verify /api/schemas endpoint returns list of schemas."""
        response = self.page.request.get("http://localhost:8000/api/schemas")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert isinstance(data, list), "Expected list of schemas"

    def test_api_corpora_endpoint(self) -> None:
        """Verify /api/corpora endpoint returns available corpora."""
        response = self.page.request.get("http://localhost:8000/api/corpora")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert "corpora" in data, "Expected 'corpora' in response"
        assert "current" in data, "Expected 'current' in response"
        assert len(data["corpora"]) > 0, "Expected at least one corpus"


class TestUserWorkflows:
    """Tests verifying primary user workflows."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_play_pause_workflow(self) -> None:
        """Verify user can play/pause simulation via API."""
        # Pause the simulation
        response = self.page.request.post("http://localhost:8000/api/world/pause")
        assert response.ok, f"Pause failed: {response.status}"

        # Check world state shows paused
        world_response = self.page.request.get("http://localhost:8000/api/world")
        world_data = world_response.json()
        assert world_data["paused"] is True, "Expected simulation to be paused"

        # Resume the simulation
        response = self.page.request.post("http://localhost:8000/api/world/play")
        assert response.ok, f"Play failed: {response.status}"

        # Check world state shows running
        world_response = self.page.request.get("http://localhost:8000/api/world")
        world_data = world_response.json()
        assert world_data["paused"] is False, "Expected simulation to be running"

    def test_speed_control_workflow(self) -> None:
        """Verify user can change simulation speed via API."""
        # Set speed to 2.0
        response = self.page.request.post("http://localhost:8000/api/world/speed?speed=2.0")
        assert response.ok, f"Speed change failed: {response.status}"

        # Check world state shows new speed
        world_response = self.page.request.get("http://localhost:8000/api/world")
        world_data = world_response.json()
        assert world_data["speed"] == 2.0, f"Expected speed 2.0, got {world_data['speed']}"

        # Reset to 1.0
        self.page.request.post("http://localhost:8000/api/world/speed?speed=1.0")

    def test_world_reset_workflow(self) -> None:
        """Verify user can reset the world via API."""
        # Let simulation run briefly
        self.page.request.post("http://localhost:8000/api/world/play")
        self.page.wait_for_timeout(500)

        # Reset world
        response = self.page.request.post("http://localhost:8000/api/world/reset")
        assert response.ok, f"Reset failed: {response.status}"

        data = response.json()
        assert data["success"] is True, f"Expected success: {data}"

    def test_corpus_switching_workflow(self) -> None:
        """Verify user can switch between corpora via API."""
        # Get available corpora
        corpora_response = self.page.request.get("http://localhost:8000/api/corpora")
        corpora_data = corpora_response.json()
        assert len(corpora_data["corpora"]) >= 2, "Expected at least 2 corpora"

        # Get a different corpus ID
        current = corpora_data["current"]
        other_corpus = next(
            (c["id"] for c in corpora_data["corpora"] if c["id"] != current),
            None,
        )
        assert other_corpus is not None, "Expected to find another corpus"

        # Switch to the other corpus
        response = self.page.request.post(
            f"http://localhost:8000/api/world/load_corpus?corpus_name={other_corpus}"
        )
        assert response.ok, f"Corpus switch failed: {response.status}"

        # Verify switch
        corpora_response = self.page.request.get("http://localhost:8000/api/corpora")
        corpora_data = corpora_response.json()
        assert corpora_data["current"] == other_corpus, "Corpus did not switch"

        # Switch back
        self.page.request.post(f"http://localhost:8000/api/world/load_corpus?corpus_name={current}")


class TestNetworkRequests:
    """Tests verifying all network requests succeed."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_no_failed_network_requests(self) -> None:
        """Verify frontend makes no failed network requests."""
        failed_requests: list[str] = []

        def handle_response(response: object) -> None:
            status = getattr(response, "status", 0)
            url = getattr(response, "url", "")
            if status >= 400:
                failed_requests.append(f"{status}: {url}")

        self.page.on("response", handle_response)

        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for all requests to complete
        self.page.wait_for_timeout(3000)

        # Filter out expected 404 (e.g., favicon)
        actual_failures = [r for r in failed_requests if "favicon" not in r]
        assert len(actual_failures) == 0, f"Failed network requests: {actual_failures}"


class TestConsoleOutput:
    """Tests verifying browser console output."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_no_console_errors(self) -> None:
        """Verify no console errors during normal operation."""
        errors: list[str] = []

        def handle_console(msg: object) -> None:
            msg_type = getattr(msg, "type", "")
            text = str(msg)
            if msg_type == "error":
                errors.append(text)

        self.page.on("console", handle_console)

        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for initialization
        self.page.wait_for_timeout(3000)

        assert len(errors) == 0, f"Console errors found: {errors}"

    def test_expected_console_messages(self) -> None:
        """Verify expected console messages appear."""
        console_messages: list[str] = []

        def handle_console(msg: object) -> None:
            console_messages.append(str(msg))

        self.page.on("console", handle_console)

        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for initialization
        self.page.wait_for_timeout(3000)

        # Check for expected initialization messages
        has_init = any("LoopEngine Visualizer initialized" in m for m in console_messages)
        has_frames_ws = any("Frames WebSocket connected" in m for m in console_messages)
        has_control_ws = any("Control WebSocket connected" in m for m in console_messages)

        assert has_init, f"Expected initialization message. Got: {console_messages[:5]}"
        assert has_frames_ws, f"Expected frames WebSocket message. Got: {console_messages[:5]}"
        assert has_control_ws, f"Expected control WebSocket message. Got: {console_messages[:5]}"
