"""Tests for the FastAPI server application."""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from loopengine.server.app import (
    ConnectionManager,
    GAJobManager,
    GAJobStatus,
    SimulationState,
    _frame_to_dict,
    app,
    get_sim_state,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sim_state() -> SimulationState:
    """Create a fresh simulation state for testing."""
    return SimulationState()


class TestSimulationState:
    """Tests for SimulationState class."""

    def test_initial_state(self, sim_state: SimulationState) -> None:
        """Test initial simulation state values."""
        assert sim_state.paused is True
        assert sim_state.speed == 1.0
        assert sim_state.world is not None
        assert sim_state.world.tick == 0

    def test_pause_toggle(self, sim_state: SimulationState) -> None:
        """Test pause/unpause functionality."""
        assert sim_state.paused is True
        sim_state.paused = False
        assert sim_state.paused is False
        sim_state.paused = True
        assert sim_state.paused is True

    def test_speed_setting(self, sim_state: SimulationState) -> None:
        """Test speed multiplier setting with clamping."""
        # Normal value
        sim_state.speed = 2.0
        assert sim_state.speed == 2.0

        # Clamped to max (10.0)
        sim_state.speed = 15.0
        assert sim_state.speed == 10.0

        # Clamped to min (0.1)
        sim_state.speed = 0.01
        assert sim_state.speed == 0.1

    def test_tick_advances_world(self, sim_state: SimulationState) -> None:
        """Test that tick advances world state."""
        initial_tick = sim_state.world.tick
        sim_state.tick()
        assert sim_state.world.tick == initial_tick + 1
        assert sim_state.latest_frame is not None

    def test_reset_restores_initial_state(self, sim_state: SimulationState) -> None:
        """Test that reset restores world to initial state."""
        # Advance the simulation
        for _ in range(5):
            sim_state.tick()
        assert sim_state.world.tick > 0

        # Reset
        sim_state.reset()
        assert sim_state.world.tick == 0
        assert sim_state.latest_frame is None

    def test_load_corpus_sandwich_shop(self, sim_state: SimulationState) -> None:
        """Test loading sandwich shop corpus."""
        sim_state.tick()
        sim_state.load_corpus("sandwich_shop")
        assert sim_state.world.tick == 0
        assert len(sim_state.world.agents) == 3  # Maria, Tom, Alex

    def test_load_corpus_unknown_raises(self, sim_state: SimulationState) -> None:
        """Test that loading unknown corpus raises ValueError."""
        with pytest.raises(ValueError, match="Unknown corpus"):
            sim_state.load_corpus("nonexistent_corpus")

    def test_thread_safety(self, sim_state: SimulationState) -> None:
        """Test thread-safe access to state."""
        results = []

        def reader_thread() -> None:
            for _ in range(100):
                _ = sim_state.world.tick
                _ = sim_state.paused
                results.append("read")

        def writer_thread() -> None:
            for _ in range(100):
                sim_state.tick()
                results.append("tick")

        threads = [
            threading.Thread(target=reader_thread),
            threading.Thread(target=writer_thread),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all operations completed
        assert len(results) == 200

    def test_start_stop_simulation(self, sim_state: SimulationState) -> None:
        """Test starting and stopping simulation thread."""
        sim_state.paused = False  # Allow ticks
        sim_state.start()
        time.sleep(0.15)  # Let it run for a bit
        sim_state.stop()
        # Simulation should have advanced
        assert sim_state.world.tick > 0

    def test_start_when_already_running(self, sim_state: SimulationState) -> None:
        """Test that starting when already running is a no-op."""
        sim_state.start()
        sim_state.start()  # Should not raise
        sim_state.stop()


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    def test_initial_state(self) -> None:
        """Test initial connection manager state."""
        manager = ConnectionManager()
        assert len(manager.frame_connections) == 0
        assert len(manager.control_connections) == 0


class TestFrameToDict:
    """Tests for _frame_to_dict function."""

    def test_converts_frame_correctly(self, sim_state: SimulationState) -> None:
        """Test that frames are converted to JSON-serializable dicts."""
        sim_state.tick()
        frame = sim_state.latest_frame
        assert frame is not None

        frame_dict = _frame_to_dict(frame)

        assert "tick" in frame_dict
        assert "time" in frame_dict
        assert "agents" in frame_dict
        assert "links" in frame_dict
        assert "particles" in frame_dict
        assert "label_regions" in frame_dict

        # Should be JSON-serializable
        json_str = json.dumps(frame_dict)
        assert json_str is not None


class TestRESTEndpoints:
    """Tests for REST API endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "healthy"}

    def test_get_world(self, client: TestClient) -> None:
        """Test GET /api/world endpoint."""
        response = client.get("/api/world")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "tick" in data
        assert "time" in data
        assert "speed" in data
        assert "paused" in data
        assert "agent_count" in data
        assert "link_count" in data
        assert "particle_count" in data

    def test_get_agents(self, client: TestClient) -> None:
        """Test GET /api/agents endpoint."""
        response = client.get("/api/agents")
        assert response.status_code == status.HTTP_200_OK
        agents = response.json()
        assert isinstance(agents, list)
        assert len(agents) == 3  # Maria, Tom, Alex

        # Check agent structure
        agent_ids = {a["id"] for a in agents}
        assert "maria" in agent_ids
        assert "tom" in agent_ids
        assert "alex" in agent_ids

        for agent in agents:
            assert "id" in agent
            assert "name" in agent
            assert "role" in agent
            assert "genome" in agent
            assert "labels" in agent
            assert "x" in agent
            assert "y" in agent
            assert "loop_phase" in agent
            assert "input_buffer_depth" in agent

    def test_get_agent_by_id(self, client: TestClient) -> None:
        """Test GET /api/agents/{agent_id} endpoint."""
        response = client.get("/api/agents/maria")
        assert response.status_code == status.HTTP_200_OK
        agent = response.json()
        assert agent["id"] == "maria"
        assert agent["name"] == "Maria"
        assert agent["role"] == "owner"

    def test_get_agent_not_found(self, client: TestClient) -> None:
        """Test GET /api/agents/{agent_id} with nonexistent agent."""
        response = client.get("/api/agents/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_get_links(self, client: TestClient) -> None:
        """Test GET /api/links endpoint."""
        response = client.get("/api/links")
        assert response.status_code == status.HTTP_200_OK
        links = response.json()
        assert isinstance(links, list)
        assert len(links) == 6  # Sandwich shop has 6 links

        for link in links:
            assert "id" in link
            assert "source_id" in link
            assert "dest_id" in link
            assert "link_type" in link
            assert "properties" in link

    def test_get_schemas(self, client: TestClient) -> None:
        """Test GET /api/schemas endpoint."""
        response = client.get("/api/schemas")
        assert response.status_code == status.HTTP_200_OK
        schemas = response.json()
        assert isinstance(schemas, list)
        # Sandwich shop has no explicit schemas, so may be empty

    def test_reset_world(self, client: TestClient) -> None:
        """Test POST /api/world/reset endpoint."""
        response = client.post("/api/world/reset")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "reset" in data["message"].lower()

    def test_load_corpus(self, client: TestClient) -> None:
        """Test POST /api/world/load_corpus endpoint."""
        response = client.post("/api/world/load_corpus?corpus_name=sandwich_shop")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True

    def test_load_corpus_unknown(self, client: TestClient) -> None:
        """Test POST /api/world/load_corpus with unknown corpus."""
        response = client.post("/api/world/load_corpus?corpus_name=unknown")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_pause_simulation(self, client: TestClient) -> None:
        """Test POST /api/world/pause endpoint."""
        response = client.post("/api/world/pause")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "paused" in data["message"].lower()

    def test_play_simulation(self, client: TestClient) -> None:
        """Test POST /api/world/play endpoint."""
        response = client.post("/api/world/play")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "playing" in data["message"].lower()

    def test_set_speed(self, client: TestClient) -> None:
        """Test POST /api/world/speed endpoint."""
        response = client.post("/api/world/speed?speed=2.0")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "2.0" in data["message"]

    def test_set_speed_clamped(self, client: TestClient) -> None:
        """Test POST /api/world/speed with value that gets clamped."""
        response = client.post("/api/world/speed?speed=100.0")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        # Should be clamped to 10.0
        assert "10.0" in data["message"]


class TestWebSocketFrames:
    """Tests for WebSocket frame streaming."""

    def test_websocket_frames_connection(self, client: TestClient) -> None:
        """Test WebSocket frames connection and frame reception."""
        with client.websocket_connect("/ws/frames") as websocket:
            # Receive one frame
            data = websocket.receive_json()

            # Should always receive a dict (either real frame or placeholder)
            assert isinstance(data, dict)
            # Should have standard frame fields
            assert "tick" in data
            assert "agents" in data

    def test_websocket_frames_structure(self, client: TestClient) -> None:
        """Test that frames have correct structure after tick."""
        # First trigger a tick to generate a real frame
        sim = get_sim_state()
        sim.tick()

        with client.websocket_connect("/ws/frames") as websocket:
            data = websocket.receive_json()

            # Should have frame structure
            assert "tick" in data
            assert "time" in data
            assert "agents" in data
            assert "links" in data
            assert "particles" in data
            assert "label_regions" in data

            # After a tick, tick should be >= 0
            assert data["tick"] >= 0


class TestWebSocketControl:
    """Tests for WebSocket control endpoint."""

    def test_websocket_control_play(self, client: TestClient) -> None:
        """Test play command via WebSocket."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "play"})
            response = websocket.receive_json()
            assert response["success"] is True
            assert "playing" in response["message"].lower()

    def test_websocket_control_pause(self, client: TestClient) -> None:
        """Test pause command via WebSocket."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "pause"})
            response = websocket.receive_json()
            assert response["success"] is True
            assert "paused" in response["message"].lower()

    def test_websocket_control_set_speed(self, client: TestClient) -> None:
        """Test set_speed command via WebSocket."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "set_speed", "speed": 2.5})
            response = websocket.receive_json()
            assert response["success"] is True
            assert "2.5" in response["message"]

    def test_websocket_control_set_speed_invalid(self, client: TestClient) -> None:
        """Test set_speed command with invalid value."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "set_speed", "speed": "not_a_number"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "invalid" in response["message"].lower()

    def test_websocket_control_reset(self, client: TestClient) -> None:
        """Test reset command via WebSocket."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "reset"})
            response = websocket.receive_json()
            assert response["success"] is True
            assert "reset" in response["message"].lower()

    def test_websocket_control_unknown_command(self, client: TestClient) -> None:
        """Test unknown command via WebSocket."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "unknown_cmd"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "unknown" in response["message"].lower()

    def test_websocket_control_case_insensitive(self, client: TestClient) -> None:
        """Test that commands are case insensitive."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "PLAY"})
            response = websocket.receive_json()
            assert response["success"] is True

            websocket.send_json({"type": "Pause"})
            response = websocket.receive_json()
            assert response["success"] is True


class TestRouterIntegration:
    """Test that existing routers are integrated."""

    def test_behaviors_router_available(self, client: TestClient) -> None:
        """Test that behaviors router is accessible."""
        # Just verify the path exists (will fail without valid domain)
        response = client.post(
            "/api/v1/behaviors/generate",
            json={"domain_id": "test", "agent_type": "test"},
        )
        # Should get 404 (domain not found) not 404 (path not found)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "domain" in response.json().get("detail", "").lower()

    def test_domains_router_available(self, client: TestClient) -> None:
        """Test that domains router is accessible."""
        response = client.get("/api/v1/domains/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "domain" in response.json().get("detail", "").lower()


class TestGAJobManager:
    """Tests for GAJobManager class."""

    def test_create_job_valid_role(self) -> None:
        """Test creating a job with valid role."""
        manager = GAJobManager()
        job_id = manager.create_job(role="sandwich_maker", generations=5, population_size=10)
        assert job_id is not None
        assert len(job_id) > 0

        job = manager.get_job(job_id)
        assert job is not None
        assert job.role == "sandwich_maker"
        assert job.generations == 5
        assert job.population_size == 10

    def test_create_job_invalid_role(self) -> None:
        """Test creating a job with invalid role raises ValueError."""
        manager = GAJobManager()
        with pytest.raises(ValueError, match="Unknown role"):
            manager.create_job(role="invalid_role")

    def test_get_job_not_found(self) -> None:
        """Test getting a non-existent job returns None."""
        manager = GAJobManager()
        assert manager.get_job("nonexistent") is None

    def test_job_runs_to_completion(self) -> None:
        """Test that a job eventually completes."""
        manager = GAJobManager()
        job_id = manager.create_job(role="sandwich_maker", generations=2, population_size=10)

        # Wait for job to complete (with timeout)
        max_wait = 30  # seconds
        wait_interval = 0.5
        elapsed = 0.0

        while elapsed < max_wait:
            job = manager.get_job(job_id)
            if job and job.status in (GAJobStatus.COMPLETED, GAJobStatus.FAILED):
                break
            time.sleep(wait_interval)
            elapsed += wait_interval

        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == GAJobStatus.COMPLETED
        assert job.current_generation == 2
        assert job.best_genome  # Should have a best genome
        assert job.best_fitness != float("-inf")  # Should have computed fitness

    def test_all_valid_roles(self) -> None:
        """Test that all valid roles can create jobs."""
        manager = GAJobManager()
        for role in ["sandwich_maker", "cashier", "owner"]:
            job_id = manager.create_job(role=role, generations=1, population_size=10)
            job = manager.get_job(job_id)
            assert job is not None
            assert job.role == role


class TestGAEndpoints:
    """Tests for GA REST API endpoints."""

    def test_ga_run_valid_request(self, client: TestClient) -> None:
        """Test POST /api/ga/run with valid request."""
        response = client.post(
            "/api/ga/run",
            json={"role": "sandwich_maker", "generations": 2, "population_size": 10},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0
        assert "message" in data
        assert "sandwich_maker" in data["message"]

    def test_ga_run_invalid_role(self, client: TestClient) -> None:
        """Test POST /api/ga/run with invalid role."""
        response = client.post(
            "/api/ga/run",
            json={"role": "invalid_role", "generations": 10, "population_size": 20},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unknown role" in response.json()["detail"]

    def test_ga_run_default_parameters(self, client: TestClient) -> None:
        """Test POST /api/ga/run with only role specified."""
        response = client.post(
            "/api/ga/run",
            json={"role": "cashier"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "job_id" in data

    def test_ga_status_valid_job(self, client: TestClient) -> None:
        """Test GET /api/ga/status/{job_id} for an existing job."""
        # First create a job
        run_response = client.post(
            "/api/ga/run",
            json={"role": "sandwich_maker", "generations": 2, "population_size": 10},
        )
        job_id = run_response.json()["job_id"]

        # Get status
        status_response = client.get(f"/api/ga/status/{job_id}")
        assert status_response.status_code == status.HTTP_200_OK
        data = status_response.json()

        assert data["job_id"] == job_id
        assert data["role"] == "sandwich_maker"
        assert data["total_generations"] == 2
        assert data["status"] in ["pending", "running", "completed", "failed"]
        assert "best_genome" in data
        # best_fitness can be None (when not yet computed) or a float
        assert "best_fitness" in data

    def test_ga_status_not_found(self, client: TestClient) -> None:
        """Test GET /api/ga/status/{job_id} for non-existent job."""
        response = client.get("/api/ga/status/nonexistent-job-id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_ga_run_and_poll_to_completion(self, client: TestClient) -> None:
        """Test full workflow: run GA and poll until completion."""
        # Start the GA run
        run_response = client.post(
            "/api/ga/run",
            json={"role": "owner", "generations": 2, "population_size": 10},
        )
        assert run_response.status_code == status.HTTP_200_OK
        job_id = run_response.json()["job_id"]

        # Poll until completion
        max_wait = 30  # seconds
        wait_interval = 0.5
        elapsed = 0.0
        final_status = None

        while elapsed < max_wait:
            status_response = client.get(f"/api/ga/status/{job_id}")
            assert status_response.status_code == status.HTTP_200_OK
            data = status_response.json()

            if data["status"] in ["completed", "failed"]:
                final_status = data
                break

            time.sleep(wait_interval)
            elapsed += wait_interval

        assert final_status is not None
        assert final_status["status"] == "completed"
        assert final_status["current_generation"] == 2
        assert len(final_status["best_genome"]) > 0
        assert final_status["best_fitness"] is not None
        # Should have stats history
        assert len(final_status["stats_history"]) == 2

    def test_ga_run_all_roles(self, client: TestClient) -> None:
        """Test that all valid roles can be run."""
        for role in ["sandwich_maker", "cashier", "owner"]:
            response = client.post(
                "/api/ga/run",
                json={"role": role, "generations": 1, "population_size": 10},
            )
            assert response.status_code == status.HTTP_200_OK
            assert "job_id" in response.json()


class TestWebSocketGACommands:
    """Tests for GA commands via WebSocket control endpoint."""

    def test_start_ga_command(self, client: TestClient) -> None:
        """Test start_ga command via WebSocket."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "sandwich_maker",
                    "generations": 2,
                    "population_size": 10,
                }
            )
            response = websocket.receive_json()
            assert response["success"] is True
            assert "job_id" in response
            assert "started" in response["message"].lower()

    def test_start_ga_missing_role(self, client: TestClient) -> None:
        """Test start_ga command without role."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "start_ga"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "role" in response["message"].lower()

    def test_start_ga_invalid_role(self, client: TestClient) -> None:
        """Test start_ga command with invalid role."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "invalid_role",
                }
            )
            response = websocket.receive_json()
            assert response["success"] is False
            assert "unknown role" in response["message"].lower()

    def test_start_ga_invalid_generations(self, client: TestClient) -> None:
        """Test start_ga command with invalid generations."""
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

    def test_start_ga_invalid_population_size(self, client: TestClient) -> None:
        """Test start_ga command with invalid population size."""
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

    def test_stop_ga_command_no_running_job(self, client: TestClient) -> None:
        """Test stop_ga command when no job is running."""
        # Use a non-existent job_id to ensure we get the "not found" response
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "stop_ga", "job_id": "nonexistent-test-job-id"})
            response = websocket.receive_json()
            assert response["success"] is False
            assert "not" in response["message"].lower()

    def test_stop_ga_with_job_id_not_found(self, client: TestClient) -> None:
        """Test stop_ga command with non-existent job_id."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "stop_ga",
                    "job_id": "nonexistent-job",
                }
            )
            response = websocket.receive_json()
            assert response["success"] is False

    def test_get_ga_status_no_running_job(self, client: TestClient) -> None:
        """Test get_ga_status response format without specifying job_id."""
        # Note: We test the response format, not the "no job running" case
        # because other tests may leave jobs running. The get_ga_status command
        # without job_id returns the running job or success: True with status: None
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json({"type": "get_ga_status"})
            response = websocket.receive_json()
            assert response["success"] is True
            # Status is either None (no running job) or a valid status string
            assert response["status"] is None or response["status"] in [
                "pending",
                "running",
                "completed",
                "failed",
            ]

    def test_get_ga_status_job_not_found(self, client: TestClient) -> None:
        """Test get_ga_status with non-existent job_id."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "get_ga_status",
                    "job_id": "nonexistent-job",
                }
            )
            response = websocket.receive_json()
            assert response["success"] is False
            assert "not found" in response["message"].lower()

    def test_get_ga_status_for_specific_job(self, client: TestClient) -> None:
        """Test get_ga_status with specific job_id."""
        with client.websocket_connect("/ws/control") as websocket:
            # Start a GA job
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "cashier",
                    "generations": 2,
                    "population_size": 10,
                }
            )
            start_response = websocket.receive_json()
            assert start_response["success"] is True
            job_id = start_response["job_id"]

            # Get status for the job
            websocket.send_json(
                {
                    "type": "get_ga_status",
                    "job_id": job_id,
                }
            )
            status_response = websocket.receive_json()
            assert status_response["success"] is True
            assert status_response["job_id"] == job_id
            assert status_response["role"] == "cashier"
            assert "status" in status_response
            assert status_response["total_generations"] == 2

    def test_start_ga_and_stop(self, client: TestClient) -> None:
        """Test starting GA and then stopping it."""
        with client.websocket_connect("/ws/control") as websocket:
            # Start a longer GA job
            websocket.send_json(
                {
                    "type": "start_ga",
                    "role": "owner",
                    "generations": 50,  # More generations to give time to stop
                    "population_size": 10,
                }
            )
            start_response = websocket.receive_json()
            assert start_response["success"] is True
            job_id = start_response["job_id"]

            # Wait briefly for job to start running
            time.sleep(0.5)

            # Stop the job
            websocket.send_json(
                {
                    "type": "stop_ga",
                    "job_id": job_id,
                }
            )
            stop_response = websocket.receive_json()
            # May succeed or fail depending on timing
            # (job might have completed quickly)
            assert "message" in stop_response

    def test_ga_commands_case_insensitive(self, client: TestClient) -> None:
        """Test that GA commands are case insensitive."""
        with client.websocket_connect("/ws/control") as websocket:
            websocket.send_json(
                {
                    "type": "START_GA",
                    "role": "sandwich_maker",
                    "generations": 1,
                    "population_size": 10,
                }
            )
            response = websocket.receive_json()
            assert response["success"] is True

            websocket.send_json({"type": "GET_GA_STATUS"})
            response = websocket.receive_json()
            assert response["success"] is True


class TestGAJobManagerStopFunctionality:
    """Tests for GAJobManager stop functionality."""

    def test_stop_job(self) -> None:
        """Test stopping a GA job."""
        manager = GAJobManager()
        job_id = manager.create_job(
            role="sandwich_maker",
            generations=100,  # Many generations
            population_size=10,
        )

        # Wait for job to start running
        max_wait = 5.0
        elapsed = 0.0
        while elapsed < max_wait:
            job = manager.get_job(job_id)
            if job and job.status == GAJobStatus.RUNNING:
                break
            time.sleep(0.1)
            elapsed += 0.1

        # Stop the job
        stopped = manager.stop_job(job_id)
        assert stopped is True

        # Wait for job to stop
        elapsed = 0.0
        while elapsed < max_wait:
            job = manager.get_job(job_id)
            if job and job.status == GAJobStatus.COMPLETED:
                break
            time.sleep(0.1)
            elapsed += 0.1

        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == GAJobStatus.COMPLETED
        assert "stopped" in job.error_message.lower()

    def test_stop_nonexistent_job(self) -> None:
        """Test stopping a non-existent job returns False."""
        manager = GAJobManager()
        assert manager.stop_job("nonexistent") is False

    def test_stop_completed_job(self) -> None:
        """Test stopping an already completed job returns False."""
        manager = GAJobManager()
        job_id = manager.create_job(
            role="sandwich_maker",
            generations=1,
            population_size=10,
        )

        # Wait for completion
        max_wait = 30.0
        elapsed = 0.0
        while elapsed < max_wait:
            job = manager.get_job(job_id)
            if job and job.status == GAJobStatus.COMPLETED:
                break
            time.sleep(0.5)
            elapsed += 0.5

        # Try to stop completed job
        stopped = manager.stop_job(job_id)
        assert stopped is False

    def test_get_running_job(self) -> None:
        """Test getting the running job."""
        manager = GAJobManager()
        # Initially no running job
        assert manager.get_running_job() is None

        # Start a job
        job_id = manager.create_job(
            role="cashier",
            generations=100,
            population_size=10,
        )

        # Wait for it to start running
        max_wait = 5.0
        elapsed = 0.0
        while elapsed < max_wait:
            running_job = manager.get_running_job()
            if running_job is not None:
                break
            time.sleep(0.1)
            elapsed += 0.1

        running_job = manager.get_running_job()
        assert running_job is not None
        assert running_job.job_id == job_id

        # Stop the job
        manager.stop_job(job_id)
