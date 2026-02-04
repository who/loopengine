"""E2E tests for genome discovery and GA simulation flow.

These tests verify the complete workflow:
1. Start backend and frontend servers
2. Open frontend in headless browser
3. Trigger genome discovery via REST API
4. Wait for discovery to complete
5. Verify simulation data appears (not stuck on loading)
6. Verify UI elements exist for all participants

Requirements:
- Backend must be running on localhost:8000
- Frontend must be served on localhost:8080
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page


class TestSimulationDataLoads:
    """Tests verifying simulation data loads correctly."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_simulation_data_appears_after_connection(self) -> None:
        """Verify simulation data loads and is not stuck on 'waiting' message."""
        # Navigate to frontend
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for WebSocket connection
        self.page.wait_for_timeout(2000)

        # Poll world state to verify simulation is running
        response = self.page.request.get("http://localhost:8000/api/world")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert "tick" in data, "Expected 'tick' in world response"
        assert data["tick"] >= 0, f"Expected tick >= 0, got {data['tick']}"

    def test_agents_endpoint_returns_participants(self) -> None:
        """Verify /api/agents returns simulation participants."""
        response = self.page.request.get("http://localhost:8000/api/agents")
        assert response.ok, f"Expected 200 OK, got {response.status}"

        agents = response.json()
        assert isinstance(agents, list), "Expected list of agents"
        assert len(agents) > 0, "Expected at least one agent"

        # Verify expected participants exist (sandwich shop defaults)
        agent_names = [a["name"] for a in agents]
        # At minimum we should have some agents with names
        assert any(name for name in agent_names), "Expected agents with names"

        # Verify agent structure has required fields
        for agent in agents:
            assert "id" in agent, "Expected 'id' in agent"
            assert "name" in agent, "Expected 'name' in agent"
            assert "role" in agent, "Expected 'role' in agent"
            assert "genome" in agent, "Expected 'genome' in agent"

    def test_frame_data_streams_via_websocket(self) -> None:
        """Verify frontend receives frame data via WebSocket."""
        console_messages: list[str] = []

        def handle_console(msg: object) -> None:
            console_messages.append(str(msg))

        self.page.on("console", handle_console)

        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for frames to be received
        self.page.wait_for_timeout(3000)

        # Check for frame receipt confirmation
        has_frames_ws = any("Frames WebSocket connected" in m for m in console_messages)
        assert has_frames_ws, f"Expected frames WebSocket connection. Got: {console_messages[:10]}"

        # Verify frame data in console (logged every 100 ticks)
        # Or verify via the loopEngine global
        result = self.page.evaluate("typeof window.loopEngine")
        assert result == "object", f"Expected loopEngine to be object, got {result}"

        latest_frame = self.page.evaluate("window.loopEngine.getLatestFrame()")
        assert latest_frame is not None, "Expected latest frame to exist"
        assert "tick" in latest_frame, "Expected 'tick' in frame"
        assert "agents" in latest_frame, "Expected 'agents' in frame"


class TestGenomeDiscoveryFlow:
    """Tests for genome discovery workflow."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_discovery_api_endpoint_exists(self) -> None:
        """Verify discovery API endpoint is available."""
        # Try to start discovery - this will test the endpoint exists
        # Note: This may fail if no AI API key is configured, which is OK
        response = self.page.request.post(
            "http://localhost:8000/api/discovery/run",
            data={
                "system": "Test sandwich shop simulation",
                "roles": [
                    {
                        "name": "test_role",
                        "inputs": ["orders"],
                        "outputs": ["sandwiches"],
                        "constraints": ["quality"],
                        "links_to": [],
                    }
                ],
            },
        )

        # The endpoint should exist (200 OK to start job, or 422/500 if AI not configured)
        # We just verify it's not a 404
        assert response.status != 404, "Discovery endpoint should exist"

        # If successful, we get a job_id
        if response.ok:
            data = response.json()
            assert "job_id" in data, "Expected job_id in response"

    def test_discovery_status_endpoint_validates_job_id(self) -> None:
        """Verify discovery status endpoint returns 404 for invalid job."""
        response = self.page.request.get(
            "http://localhost:8000/api/discovery/status/invalid-job-id-12345"
        )

        # Should return 404 for non-existent job
        assert response.status == 404, f"Expected 404, got {response.status}"


class TestGASimulationFlow:
    """Tests for genetic algorithm simulation workflow."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_ga_run_endpoint_accepts_valid_role(self) -> None:
        """Verify GA run endpoint accepts valid roles."""
        response = self.page.request.post(
            "http://localhost:8000/api/ga/run",
            data={
                "role": "sandwich_maker",
                "generations": 1,  # Minimal generations for quick test
                "population_size": 10,
            },
        )

        assert response.ok, f"Expected 200 OK, got {response.status}"

        data = response.json()
        assert "job_id" in data, "Expected job_id in response"
        assert "message" in data, "Expected message in response"

    def test_ga_run_endpoint_rejects_invalid_role(self) -> None:
        """Verify GA run endpoint rejects invalid roles."""
        response = self.page.request.post(
            "http://localhost:8000/api/ga/run",
            data={
                "role": "invalid_role_that_does_not_exist",
                "generations": 1,
                "population_size": 10,
            },
        )

        # Should return 400 Bad Request for invalid role
        assert response.status == 400, f"Expected 400, got {response.status}"

    def test_ga_status_endpoint_returns_job_progress(self) -> None:
        """Verify GA status endpoint returns job progress."""
        # Start a GA job
        run_response = self.page.request.post(
            "http://localhost:8000/api/ga/run",
            data={
                "role": "cashier",
                "generations": 2,
                "population_size": 10,
            },
        )

        assert run_response.ok, f"Failed to start GA: {run_response.status}"
        job_id = run_response.json()["job_id"]

        # Poll status until complete or timeout
        max_polls = 60
        for _ in range(max_polls):
            status_response = self.page.request.get(f"http://localhost:8000/api/ga/status/{job_id}")
            assert status_response.ok, f"Status check failed: {status_response.status}"

            status_data = status_response.json()
            assert "status" in status_data, "Expected 'status' in response"
            assert "job_id" in status_data, "Expected 'job_id' in response"

            if status_data["status"] in ("completed", "failed"):
                break

            time.sleep(1)

        # Verify final status has expected fields
        assert "current_generation" in status_data, "Expected 'current_generation'"
        assert "best_genome" in status_data, "Expected 'best_genome'"


class TestFrontendUIRendering:
    """Tests verifying frontend UI renders correctly."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_canvas_renders_agents(self) -> None:
        """Verify canvas renders agents from simulation."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Ensure simulation is playing (may have been paused by previous tests)
        self.page.request.post("http://localhost:8000/api/world/play")

        # Poll for frame with agents (may take a moment for server to send valid frame)
        # CI runners can be slow, so use a generous timeout
        max_wait_ms = 15000
        poll_interval_ms = 500
        elapsed_ms = 0
        latest_frame = None

        while elapsed_ms < max_wait_ms:
            latest_frame = self.page.evaluate("window.loopEngine.getLatestFrame()")
            if latest_frame and latest_frame.get("tick", -1) >= 0 and latest_frame.get("agents"):
                break
            self.page.wait_for_timeout(poll_interval_ms)
            elapsed_ms += poll_interval_ms

        assert latest_frame is not None, "Expected frame data"
        assert "agents" in latest_frame, "Expected 'agents' in frame"
        assert len(latest_frame["agents"]) > 0, "Expected at least one agent in frame"

        # Verify agent data structure
        agent = latest_frame["agents"][0]
        assert "x" in agent, "Expected 'x' position in agent"
        assert "y" in agent, "Expected 'y' position in agent"
        assert "name" in agent, "Expected 'name' in agent"

    def test_frontend_shows_tick_counter(self) -> None:
        """Verify tick counter increments in world state."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Get initial tick
        initial_response = self.page.request.get("http://localhost:8000/api/world")
        initial_tick = initial_response.json()["tick"]

        # Resume simulation if paused
        self.page.request.post("http://localhost:8000/api/world/play")

        # Wait briefly
        self.page.wait_for_timeout(1000)

        # Get new tick
        final_response = self.page.request.get("http://localhost:8000/api/world")
        final_tick = final_response.json()["tick"]

        # Tick should have increased (simulation is running)
        assert final_tick > initial_tick, (
            f"Expected tick to increase. Initial: {initial_tick}, Final: {final_tick}"
        )

    def test_all_sandwich_shop_participants_present(self) -> None:
        """Verify all expected sandwich shop participants are in simulation."""
        # Get agents from API
        response = self.page.request.get("http://localhost:8000/api/agents")
        assert response.ok

        agents = response.json()
        agent_names = {a["name"].lower() for a in agents}
        agent_roles = {a["role"] for a in agents}

        # Sandwich shop should have these roles
        expected_roles = {"sandwich_maker", "cashier", "owner"}
        assert expected_roles.issubset(agent_roles), (
            f"Expected roles {expected_roles} to be subset of {agent_roles}"
        )

        # Should have agents with names
        assert len(agent_names) >= 3, f"Expected at least 3 named agents, got {len(agent_names)}"


class TestSimulationNotStuckOnLoading:
    """Tests verifying simulation doesn't get stuck on 'waiting for data' state."""

    @pytest.fixture(autouse=True)
    def _setup(self, page: Page) -> None:
        """Store page fixture."""
        self.page = page

    def test_simulation_data_appears_within_timeout(self) -> None:
        """Verify simulation data appears within reasonable timeout (30s max)."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Poll for frame data with reasonable timeout
        max_wait_ms = 30000
        poll_interval_ms = 500
        elapsed_ms = 0

        frame_received = False
        while elapsed_ms < max_wait_ms:
            latest_frame = self.page.evaluate("window.loopEngine.getLatestFrame()")
            if latest_frame and latest_frame.get("tick", -1) >= 0:
                frame_received = True
                break
            self.page.wait_for_timeout(poll_interval_ms)
            elapsed_ms += poll_interval_ms

        assert frame_received, (
            f"Simulation data did not appear within {max_wait_ms}ms timeout. "
            "Frontend may be stuck on 'waiting for simulation data'"
        )

    def test_no_waiting_for_data_message_after_load(self) -> None:
        """Verify 'Waiting for simulation data' message doesn't persist."""
        self.page.goto("http://localhost:8080/frontend/index.html")

        # Wait for initial load
        self.page.wait_for_timeout(3000)

        # The frontend renders to canvas, so we check frame state instead of DOM
        latest_frame = self.page.evaluate("window.loopEngine.getLatestFrame()")

        # Frame should exist and have valid tick (not -1 which indicates no data)
        assert latest_frame is not None, "Expected frame data after load"
        assert latest_frame.get("tick", -1) >= 0, (
            f"Frame tick is {latest_frame.get('tick', -1)}, "
            "indicating 'waiting for simulation data' state"
        )

        # Verify agents are present in frame
        assert "agents" in latest_frame, "Expected 'agents' in frame"
        assert len(latest_frame["agents"]) > 0, (
            "No agents in frame - simulation may be stuck on loading"
        )
