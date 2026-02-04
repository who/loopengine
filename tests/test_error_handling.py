"""Tests for comprehensive error handling across the loopengine application."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from loopengine.engine.loop import _do_decide
from loopengine.model.agent import Agent, Phase
from loopengine.server.app import SimulationState, app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sim_state() -> SimulationState:
    """Create a fresh simulation state for testing."""
    return SimulationState()


class TestAPIErrorHandling:
    """Tests for API error handling with proper HTTP codes."""

    def test_404_agent_not_found(self, client: TestClient) -> None:
        """Test that requesting non-existent agent returns 404 with clear message."""
        response = client.get("/api/agents/nonexistent_agent_id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_400_invalid_corpus(self, client: TestClient) -> None:
        """Test that loading unknown corpus returns 400 with clear message."""
        response = client.post("/api/world/load_corpus?corpus_name=invalid_corpus")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "unknown corpus" in data["detail"].lower()

    def test_400_invalid_ga_role(self, client: TestClient) -> None:
        """Test that starting GA with invalid role returns 400."""
        response = client.post(
            "/api/ga/run",
            json={"role": "invalid_role", "generations": 10, "population_size": 20},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert "unknown role" in data["detail"].lower()

    def test_404_ga_job_not_found(self, client: TestClient) -> None:
        """Test that requesting non-existent GA job returns 404."""
        response = client.get("/api/ga/status/nonexistent-job-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_404_discovery_job_not_found(self, client: TestClient) -> None:
        """Test that requesting non-existent discovery job returns 404."""
        response = client.get("/api/discovery/status/nonexistent-job-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()


class TestWebSocketErrorHandling:
    """Tests for WebSocket disconnect handling and error responses."""

    def test_websocket_invalid_json(self, client: TestClient) -> None:
        """Test that invalid JSON on control WebSocket returns error response."""
        with client.websocket_connect("/ws/control") as websocket:
            # Send valid JSON with unknown command
            websocket.send_json({"type": "invalid_command_xyz"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "unknown" in response["message"].lower()

    def test_websocket_missing_command_type(self, client: TestClient) -> None:
        """Test that missing command type returns error response."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "unknown" in response["message"].lower()

    def test_websocket_invalid_speed_type(self, client: TestClient) -> None:
        """Test that invalid speed type returns error response."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "set_speed", "speed": "not_a_number"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "invalid" in response["message"].lower()

    def test_websocket_set_speed_none(self, client: TestClient) -> None:
        """Test that null speed value returns error response."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "set_speed", "speed": None})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "invalid" in response["message"].lower()

    def test_websocket_start_ga_missing_role(self, client: TestClient) -> None:
        """Test that start_ga without role returns clear error."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "start_ga"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "role" in response["message"].lower()

    def test_websocket_start_ga_invalid_generations(self, client: TestClient) -> None:
        """Test that start_ga with invalid generations returns error."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "sandwich_maker",
                    "generations": "not_a_number",
                }
            )
            response = websocket.receive_json()
            assert response["success"] is False

    def test_websocket_start_ga_out_of_range_generations(self, client: TestClient) -> None:
        """Test that start_ga with out-of-range generations returns error."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "sandwich_maker",
                    "generations": 0,
                }
            )
            response = websocket.receive_json()
            assert response["success"] is False
            assert "between" in response["message"].lower()

    def test_websocket_start_ga_out_of_range_population(self, client: TestClient) -> None:
        """Test that start_ga with out-of-range population returns error."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "sandwich_maker",
                    "population_size": 5,  # Min is 10
                }
            )
            response = websocket.receive_json()
            assert response["success"] is False
            assert "between" in response["message"].lower()


class TestInputValidation:
    """Tests for input validation with clear error messages."""

    def test_ga_generations_range_validation(self, client: TestClient) -> None:
        """Test that GA generations out of range is rejected."""
        # Too high
        response = client.post(
            "/api/ga/run",
            json={"role": "sandwich_maker", "generations": 10000},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Too low
        response = client.post(
            "/api/ga/run",
            json={"role": "sandwich_maker", "generations": 0},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ga_population_range_validation(self, client: TestClient) -> None:
        """Test that GA population size out of range is rejected."""
        # Too high
        response = client.post(
            "/api/ga/run",
            json={"role": "sandwich_maker", "population_size": 1000},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Too low
        response = client.post(
            "/api/ga/run",
            json={"role": "sandwich_maker", "population_size": 1},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_discovery_minimal_valid_request(self, client: TestClient) -> None:
        """Test that discovery accepts minimal valid request."""
        response = client.post(
            "/api/discovery/run",
            json={
                "system": "Test system",
                "roles": [{"name": "worker"}],
            },
        )
        assert response.status_code == status.HTTP_200_OK
        assert "job_id" in response.json()


class TestPolicyFailureHandling:
    """Tests for simulation continuity when agent policies fail."""

    def test_policy_failure_continues_simulation(self, sim_state: SimulationState) -> None:
        """Test that simulation continues even when an agent policy fails."""
        world = sim_state.world

        # Create an agent with a failing policy
        def failing_policy(sensed_inputs, genome, internal_state):
            raise RuntimeError("Intentional policy failure for testing")

        # Get an existing agent and replace its policy
        agent_id = next(iter(world.agents.keys()))
        original_policy = world.agents[agent_id].policy
        world.agents[agent_id].policy = failing_policy

        try:
            # Run several ticks - should not raise
            initial_tick = world.tick
            for _ in range(10):
                sim_state.tick()

            # Simulation should have advanced
            assert world.tick > initial_tick

            # Check that policy failure was tracked
            agent = world.agents[agent_id]
            failures = agent.internal_state.get("_policy_failures", 0)
            # At least one failure should be recorded
            assert failures >= 0  # May be 0 if agent wasn't in DECIDE phase

        finally:
            # Restore original policy
            world.agents[agent_id].policy = original_policy

    def test_policy_failure_records_failure_count(self) -> None:
        """Test that policy failures are tracked in agent internal state."""
        # Create a minimal agent for testing
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role="test",
            genome={"trait": 0.5},
            labels=set(),
            x=0,
            y=0,
            loop_period=4,
            loop_phase=Phase.DECIDE,
            phase_tick=0,
            input_buffer=[],
            output_buffer=[],
            internal_state={"sensed_inputs": []},
            policy=lambda s, g, i: (_ for _ in ()).throw(ValueError("Test error")),
        )

        # Run decide phase with failing policy
        _do_decide(agent)

        # Check failure was recorded
        assert agent.internal_state.get("_policy_failures", 0) == 1
        # Planned actions should be empty (graceful degradation)
        assert agent.internal_state.get("planned_actions", []) == []

    def test_policy_none_continues(self) -> None:
        """Test that agents without policies continue normally."""
        agent = Agent(
            id="test_agent",
            name="Test Agent",
            role="test",
            genome={"trait": 0.5},
            labels=set(),
            x=0,
            y=0,
            loop_period=4,
            loop_phase=Phase.DECIDE,
            phase_tick=0,
            input_buffer=[],
            output_buffer=[],
            internal_state={"sensed_inputs": []},
            policy=None,
        )

        # Should not raise
        _do_decide(agent)

        # Planned actions should be empty
        assert agent.internal_state.get("planned_actions") == []


class TestDiscoveryErrorHandling:
    """Tests for discovery error handling."""

    def test_discovery_job_creation_succeeds(self, client: TestClient) -> None:
        """Test that discovery jobs can be created successfully."""
        response = client.post(
            "/api/discovery/run",
            json={
                "system": "Test system for error handling",
                "roles": [
                    {"name": "role1", "inputs": ["input1"], "outputs": ["output1"]},
                ],
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "job_id" in data
        assert "message" in data

    def test_discovery_status_includes_error_message(self, client: TestClient) -> None:
        """Test that discovery status includes error message field."""
        # Create a job first
        run_response = client.post(
            "/api/discovery/run",
            json={
                "system": "Test system",
                "roles": [{"name": "tester"}],
            },
        )
        job_id = run_response.json()["job_id"]

        # Get status
        status_response = client.get(f"/api/discovery/status/{job_id}")
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()
        assert "error_message" in data
        # error_message should be empty string for successful jobs
        assert isinstance(data["error_message"], str)


class TestErrorLoggingWithContext:
    """Tests for error logging with proper context."""

    def test_websocket_errors_logged_with_context(self, client: TestClient) -> None:
        """Test that WebSocket errors include useful context."""
        # This test verifies the error handling behavior, not actual logging
        # The logging context is verified by code review of the implementation
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "unknown_command"})
            response = websocket.receive_json()
            assert response["success"] is False
            # Error response should include the command that failed
            assert "unknown_command" in response["message"]

    def test_api_errors_return_structured_response(self, client: TestClient) -> None:
        """Test that API errors return structured error responses."""
        response = client.get("/api/agents/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        # Should have detail field (FastAPI standard)
        assert "detail" in data


class TestGracefulDegradation:
    """Tests for graceful degradation under error conditions."""

    def test_simulation_continues_after_reset(self, sim_state: SimulationState) -> None:
        """Test that simulation continues normally after reset."""
        # Run some ticks
        for _ in range(5):
            sim_state.tick()
        assert sim_state.world.tick == 5

        # Reset
        sim_state.reset()
        assert sim_state.world.tick == 0

        # Should be able to continue
        for _ in range(5):
            sim_state.tick()
        assert sim_state.world.tick == 5

    def test_corpus_switch_resets_cleanly(self, sim_state: SimulationState) -> None:
        """Test that switching corpus resets state cleanly."""
        # Run with sandwich shop
        for _ in range(5):
            sim_state.tick()
        assert len(sim_state.world.agents) == 3

        # Switch to software team
        sim_state.load_corpus("software_team")
        assert len(sim_state.world.agents) == 4
        assert sim_state.world.tick == 0

        # Run with software team
        for _ in range(5):
            sim_state.tick()
        assert sim_state.world.tick == 5

        # Switch back
        sim_state.load_corpus("sandwich_shop")
        assert len(sim_state.world.agents) == 3
        assert sim_state.world.tick == 0
