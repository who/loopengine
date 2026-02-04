"""FastAPI server with REST API and WebSocket frame streaming.

Provides:
- WebSocket /ws/frames: Stream Frame objects at ~30 FPS
- WebSocket /ws/control: Receive play/pause/set_speed commands
- REST API for world state and agent information
- GA run API for genetic algorithm evolution
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from loopengine.api.behaviors import router as behaviors_router
from loopengine.api.domains import router as domains_router
from loopengine.corpora.sandwich_shop import create_world
from loopengine.engine.fitness import alex_fitness, maria_fitness, tom_fitness
from loopengine.engine.ga import GAEngine
from loopengine.engine.simulation import tick_world
from loopengine.model.genome import GenomeSchema, GenomeTrait
from loopengine.projection.projector import Frame, project

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class SimulationState:
    """Thread-safe simulation state manager.

    Manages the world state and provides synchronization between
    the background simulation thread and WebSocket/REST endpoints.
    """

    def __init__(self) -> None:
        """Initialize simulation state with default values."""
        self._world = create_world()
        self._running = False
        self._speed = 1.0  # Simulation speed multiplier
        self._paused = True  # Start paused
        self._lock = threading.Lock()
        self._frame_event = asyncio.Event()
        self._latest_frame: Frame | None = None
        self._stop_event = threading.Event()

    @property
    def world(self) -> Any:
        """Get current world state (thread-safe)."""
        with self._lock:
            return self._world

    @property
    def paused(self) -> bool:
        """Check if simulation is paused."""
        with self._lock:
            return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        """Set paused state."""
        with self._lock:
            self._paused = value

    @property
    def speed(self) -> float:
        """Get simulation speed multiplier."""
        with self._lock:
            return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set simulation speed (clamped to 0.1-10.0)."""
        with self._lock:
            self._speed = max(0.1, min(10.0, value))

    @property
    def latest_frame(self) -> Frame | None:
        """Get the latest projected frame."""
        with self._lock:
            return self._latest_frame

    def tick(self) -> None:
        """Execute one simulation tick (thread-safe)."""
        with self._lock:
            tick_world(self._world)
            self._latest_frame = project(self._world)

    def reset(self) -> None:
        """Reset world to initial state."""
        with self._lock:
            self._world = create_world()
            self._latest_frame = None

    def load_corpus(self, corpus_name: str) -> None:
        """Load a named corpus.

        Args:
            corpus_name: Name of the corpus to load.

        Raises:
            ValueError: If corpus name is not recognized.
        """
        with self._lock:
            if corpus_name == "sandwich_shop":
                self._world = create_world()
                self._latest_frame = None
            else:
                raise ValueError(f"Unknown corpus: {corpus_name}")

    def start(self) -> None:
        """Start the background simulation thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()
        logger.info("Simulation thread started")

    def stop(self) -> None:
        """Stop the background simulation thread."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        logger.info("Simulation thread stopped")

    def _simulation_loop(self) -> None:
        """Background simulation loop running at ~30 ticks/second base rate."""
        target_fps = 30.0
        while self._running and not self._stop_event.is_set():
            if not self._paused:
                self.tick()

            # Adjust sleep time based on speed
            effective_speed = self.speed if not self._paused else 1.0
            sleep_time = 1.0 / (target_fps * effective_speed)
            self._stop_event.wait(timeout=sleep_time)


# Global simulation state
_sim_state: SimulationState | None = None


# GA Job Management


class GAJobStatus(StrEnum):
    """Status of a GA job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GAJob:
    """Represents a GA evolution job."""

    job_id: str
    role: str
    target_agent_id: str
    generations: int
    population_size: int
    status: GAJobStatus = GAJobStatus.PENDING
    current_generation: int = 0
    best_genome: dict[str, float] = field(default_factory=dict)
    best_fitness: float = float("-inf")
    error_message: str = ""
    stats_history: list[dict[str, Any]] = field(default_factory=list)


class GAJobManager:
    """Thread-safe manager for GA background jobs."""

    # Mapping from role to (target_agent_id, fitness_fn)
    ROLE_CONFIG: ClassVar[dict[str, tuple[str, Any]]] = {
        "sandwich_maker": ("tom", tom_fitness),
        "cashier": ("alex", alex_fitness),
        "owner": ("maria", maria_fitness),
    }

    def __init__(self) -> None:
        """Initialize the GA job manager."""
        self._jobs: dict[str, GAJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ga_worker")

    def create_job(
        self,
        role: str,
        generations: int = 100,
        population_size: int = 50,
    ) -> str:
        """Create a new GA job and start it in the background.

        Args:
            role: Agent role to evolve (sandwich_maker, cashier, owner).
            generations: Number of generations to run.
            population_size: Size of the population.

        Returns:
            str: Job ID for status polling.

        Raises:
            ValueError: If role is not recognized.
        """
        if role not in self.ROLE_CONFIG:
            valid_roles = ", ".join(self.ROLE_CONFIG.keys())
            msg = f"Unknown role: {role}. Valid roles: {valid_roles}"
            raise ValueError(msg)

        target_agent_id, _ = self.ROLE_CONFIG[role]

        job_id = str(uuid.uuid4())
        job = GAJob(
            job_id=job_id,
            role=role,
            target_agent_id=target_agent_id,
            generations=generations,
            population_size=population_size,
        )

        with self._lock:
            self._jobs[job_id] = job

        # Submit job to executor
        self._executor.submit(self._run_ga_job, job_id)

        return job_id

    def get_job(self, job_id: str) -> GAJob | None:
        """Get job by ID.

        Args:
            job_id: The job ID to look up.

        Returns:
            GAJob or None if not found.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def _run_ga_job(self, job_id: str) -> None:
        """Run GA evolution in background thread.

        Args:
            job_id: The job to run.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = GAJobStatus.RUNNING

        try:
            # Get role configuration
            target_agent_id, fitness_fn = self.ROLE_CONFIG[job.role]

            # Create a fresh world template for evaluation
            world_template = create_world()

            # Get the target agent to extract schema from genome keys
            target_agent = world_template.agents[target_agent_id]
            schema = GenomeSchema(role=job.role)
            for trait_name in target_agent.genome.keys():
                schema.traits[trait_name] = GenomeTrait(
                    name=trait_name,
                    description=f"Trait {trait_name} for {job.role}",
                    min_val=0.0,
                    max_val=1.0,
                )

            # Create GA engine
            ga = GAEngine(
                population_size=job.population_size,
                selection_count=max(2, job.population_size // 5),
                mutation_rate=0.1,
                mutation_magnitude=0.05,
            )
            ga.initialize_population(schema)

            # Create fitness evaluator that runs simulation
            def evaluate_genome(genome: dict[str, float]) -> float:
                """Evaluate a genome by running simulation."""
                from loopengine.engine.ga import evaluate_fitness

                return evaluate_fitness(
                    genome=genome,
                    world_template=world_template,
                    target_agent_id=target_agent_id,
                    ticks=500,  # Shorter for faster evaluation
                    fitness_fn=fitness_fn,
                )

            # Run generations
            for gen in range(job.generations):
                stats = ga.run_generation(evaluate_genome)

                # Update job progress
                with self._lock:
                    job.current_generation = gen + 1
                    job.best_genome = ga.best_genome.copy()
                    job.best_fitness = ga.best_fitness
                    job.stats_history.append(
                        {
                            "generation": stats.generation,
                            "best_fitness": stats.best_fitness,
                            "avg_fitness": stats.avg_fitness,
                            "min_fitness": stats.min_fitness,
                            "max_fitness": stats.max_fitness,
                        }
                    )

            # Mark complete
            with self._lock:
                job.status = GAJobStatus.COMPLETED

        except Exception as e:
            with self._lock:
                job.status = GAJobStatus.FAILED
                job.error_message = str(e)
            logger.exception("GA job %s failed: %s", job_id, e)


# Global GA job manager
_ga_job_manager: GAJobManager | None = None


def get_ga_job_manager() -> GAJobManager:
    """Get or create the global GA job manager."""
    global _ga_job_manager
    if _ga_job_manager is None:
        _ga_job_manager = GAJobManager()
    return _ga_job_manager


def get_sim_state() -> SimulationState:
    """Get or create the global simulation state."""
    global _sim_state
    if _sim_state is None:
        _sim_state = SimulationState()
    return _sim_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: start/stop simulation thread."""
    sim = get_sim_state()
    sim.start()
    yield
    sim.stop()


# Create FastAPI app
app = FastAPI(
    title="LoopEngine",
    description="Simulation framework and visual explorer for modeling agents",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(behaviors_router)
app.include_router(domains_router)


# Pydantic models for REST responses


class AgentResponse(BaseModel):
    """Response model for agent information."""

    id: str = Field(description="Agent ID")
    name: str = Field(description="Agent name")
    role: str = Field(description="Agent role")
    genome: dict[str, float] = Field(description="Agent genome traits")
    labels: list[str] = Field(description="Labels assigned to agent")
    x: float = Field(description="X position")
    y: float = Field(description="Y position")
    loop_phase: str = Field(description="Current OODA loop phase")
    input_buffer_depth: int = Field(description="Number of items in input buffer")


class LinkResponse(BaseModel):
    """Response model for link information."""

    id: str = Field(description="Link ID")
    source_id: str = Field(description="Source agent ID")
    dest_id: str = Field(description="Destination agent ID")
    link_type: str = Field(description="Type of link")
    properties: dict[str, Any] = Field(description="Link properties")


class WorldStateResponse(BaseModel):
    """Response model for full world state."""

    tick: int = Field(description="Current simulation tick")
    time: float = Field(description="Elapsed simulation time")
    speed: float = Field(description="Simulation speed")
    paused: bool = Field(description="Whether simulation is paused")
    agent_count: int = Field(description="Number of agents")
    link_count: int = Field(description="Number of links")
    particle_count: int = Field(description="Number of active particles")


class SchemaResponse(BaseModel):
    """Response model for genome schema."""

    role: str = Field(description="Role this schema applies to")
    traits: list[dict[str, Any]] = Field(description="Trait definitions")


class ControlCommandResponse(BaseModel):
    """Response for control commands."""

    success: bool = Field(description="Whether command succeeded")
    message: str = Field(description="Status message")


# GA API models


class GARunRequest(BaseModel):
    """Request model for starting a GA run."""

    role: str = Field(description="Agent role to evolve (sandwich_maker, cashier, owner)")
    generations: int = Field(default=100, ge=1, le=1000, description="Number of generations")
    population_size: int = Field(default=50, ge=10, le=500, description="Population size")


class GARunResponse(BaseModel):
    """Response model for GA run initiation."""

    job_id: str = Field(description="Job ID for status polling")
    message: str = Field(description="Status message")


class GAStatusResponse(BaseModel):
    """Response model for GA job status."""

    job_id: str = Field(description="Job ID")
    status: str = Field(description="Job status: pending, running, completed, failed")
    role: str = Field(description="Agent role being evolved")
    current_generation: int = Field(description="Current generation number")
    total_generations: int = Field(description="Total generations to run")
    best_fitness: float | None = Field(description="Best fitness found so far")
    best_genome: dict[str, float] = Field(description="Best genome found")
    error_message: str = Field(default="", description="Error message if failed")
    stats_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Generation stats history"
    )


# REST endpoints


@app.get("/api/world", response_model=WorldStateResponse, tags=["world"])
async def get_world() -> WorldStateResponse:
    """Get current world state summary."""
    sim = get_sim_state()
    world = sim.world
    return WorldStateResponse(
        tick=world.tick,
        time=world.time,
        speed=sim.speed,
        paused=sim.paused,
        agent_count=len(world.agents),
        link_count=len(world.links),
        particle_count=len(world.particles),
    )


@app.get("/api/agents", response_model=list[AgentResponse], tags=["agents"])
async def get_agents() -> list[AgentResponse]:
    """Get list of all agents."""
    sim = get_sim_state()
    world = sim.world
    agents = []
    for agent in world.agents.values():
        agents.append(
            AgentResponse(
                id=agent.id,
                name=agent.name,
                role=agent.role,
                genome=agent.genome,
                labels=list(agent.labels),
                x=agent.x,
                y=agent.y,
                loop_phase=agent.loop_phase.value,
                input_buffer_depth=len(agent.input_buffer),
            )
        )
    return agents


@app.get("/api/agents/{agent_id}", response_model=AgentResponse, tags=["agents"])
async def get_agent(agent_id: str) -> AgentResponse:
    """Get a specific agent by ID."""
    sim = get_sim_state()
    world = sim.world
    agent = world.agents.get(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        genome=agent.genome,
        labels=list(agent.labels),
        x=agent.x,
        y=agent.y,
        loop_phase=agent.loop_phase.value,
        input_buffer_depth=len(agent.input_buffer),
    )


@app.get("/api/links", response_model=list[LinkResponse], tags=["links"])
async def get_links() -> list[LinkResponse]:
    """Get list of all links."""
    sim = get_sim_state()
    world = sim.world
    links = []
    for link in world.links.values():
        links.append(
            LinkResponse(
                id=link.id,
                source_id=link.source_id,
                dest_id=link.dest_id,
                link_type=link.link_type.value,
                properties=link.properties,
            )
        )
    return links


@app.get("/api/schemas", response_model=list[SchemaResponse], tags=["schemas"])
async def get_schemas() -> list[SchemaResponse]:
    """Get all genome schemas."""
    sim = get_sim_state()
    world = sim.world
    schemas = []
    for role, schema in world.schemas.items():
        # Convert schema to dict format
        traits = []
        if hasattr(schema, "traits"):
            for trait in schema.traits:
                traits.append(
                    {
                        "name": trait.name,
                        "min": trait.min_value,
                        "max": trait.max_value,
                        "default": trait.default,
                    }
                )
        schemas.append(SchemaResponse(role=role, traits=traits))
    return schemas


@app.post("/api/world/reset", response_model=ControlCommandResponse, tags=["world"])
async def reset_world() -> ControlCommandResponse:
    """Reset world to initial state."""
    sim = get_sim_state()
    sim.reset()
    logger.info("World reset to initial state")
    return ControlCommandResponse(success=True, message="World reset to initial state")


@app.post("/api/world/load_corpus", response_model=ControlCommandResponse, tags=["world"])
async def load_corpus(corpus_name: str = "sandwich_shop") -> ControlCommandResponse:
    """Load a named corpus.

    Args:
        corpus_name: Name of the corpus to load (default: sandwich_shop).
    """
    sim = get_sim_state()
    try:
        sim.load_corpus(corpus_name)
        logger.info("Loaded corpus: %s", corpus_name)
        return ControlCommandResponse(success=True, message=f"Loaded corpus: {corpus_name}")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@app.post("/api/world/pause", response_model=ControlCommandResponse, tags=["world"])
async def pause_simulation() -> ControlCommandResponse:
    """Pause the simulation."""
    sim = get_sim_state()
    sim.paused = True
    return ControlCommandResponse(success=True, message="Simulation paused")


@app.post("/api/world/play", response_model=ControlCommandResponse, tags=["world"])
async def play_simulation() -> ControlCommandResponse:
    """Resume the simulation."""
    sim = get_sim_state()
    sim.paused = False
    return ControlCommandResponse(success=True, message="Simulation playing")


@app.post("/api/world/speed", response_model=ControlCommandResponse, tags=["world"])
async def set_speed(speed: float = 1.0) -> ControlCommandResponse:
    """Set simulation speed multiplier.

    Args:
        speed: Speed multiplier (0.1-10.0, default: 1.0).
    """
    sim = get_sim_state()
    sim.speed = speed
    return ControlCommandResponse(success=True, message=f"Speed set to {sim.speed}")


# GA endpoints


@app.post("/api/ga/run", response_model=GARunResponse, tags=["ga"])
async def start_ga_run(request: GARunRequest) -> GARunResponse:
    """Start a GA evolution run in the background.

    Runs the genetic algorithm asynchronously to evolve the specified role.
    Returns a job_id that can be used to poll for status via GET /api/ga/status/{job_id}.

    Args:
        request: GA run request with role, generations, and population_size.

    Returns:
        GARunResponse with job_id for status polling.

    Raises:
        HTTPException: If role is not valid.
    """
    ga_manager = get_ga_job_manager()
    try:
        job_id = ga_manager.create_job(
            role=request.role,
            generations=request.generations,
            population_size=request.population_size,
        )
        logger.info("Started GA job %s for role=%s", job_id, request.role)
        return GARunResponse(
            job_id=job_id,
            message=f"GA evolution started for role '{request.role}'",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@app.get("/api/ga/status/{job_id}", response_model=GAStatusResponse, tags=["ga"])
async def get_ga_status(job_id: str) -> GAStatusResponse:
    """Get status of a GA evolution job.

    Poll this endpoint to check progress and retrieve results when complete.

    Args:
        job_id: The job ID returned from POST /api/ga/run.

    Returns:
        GAStatusResponse with current status, progress, and best genome.

    Raises:
        HTTPException: If job_id is not found.
    """
    ga_manager = get_ga_job_manager()
    job = ga_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GA job '{job_id}' not found",
        )

    # Handle -inf which is not JSON serializable
    best_fitness: float | None = job.best_fitness
    if math.isinf(best_fitness):
        best_fitness = None if best_fitness < 0 else float("inf")

    return GAStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        role=job.role,
        current_generation=job.current_generation,
        total_generations=job.generations,
        best_fitness=best_fitness,
        best_genome=job.best_genome,
        error_message=job.error_message,
        stats_history=job.stats_history,
    )


# WebSocket connections management


class ConnectionManager:
    """Manage WebSocket connections for frame streaming."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.frame_connections: list[WebSocket] = []
        self.control_connections: list[WebSocket] = []

    async def connect_frames(self, websocket: WebSocket) -> None:
        """Accept a frames WebSocket connection."""
        await websocket.accept()
        self.frame_connections.append(websocket)
        logger.info("Frame client connected, total: %d", len(self.frame_connections))

    async def connect_control(self, websocket: WebSocket) -> None:
        """Accept a control WebSocket connection."""
        await websocket.accept()
        self.control_connections.append(websocket)
        logger.info("Control client connected, total: %d", len(self.control_connections))

    def disconnect_frames(self, websocket: WebSocket) -> None:
        """Remove a frames WebSocket connection."""
        if websocket in self.frame_connections:
            self.frame_connections.remove(websocket)
        logger.info("Frame client disconnected, remaining: %d", len(self.frame_connections))

    def disconnect_control(self, websocket: WebSocket) -> None:
        """Remove a control WebSocket connection."""
        if websocket in self.control_connections:
            self.control_connections.remove(websocket)
        logger.info("Control client disconnected, remaining: %d", len(self.control_connections))

    async def broadcast_frame(self, frame_data: dict[str, Any]) -> None:
        """Broadcast frame to all connected clients."""
        disconnected = []
        for connection in self.frame_connections:
            try:
                await connection.send_json(frame_data)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect_frames(conn)


# Global connection manager
manager = ConnectionManager()


def _frame_to_dict(frame: Frame) -> dict[str, Any]:
    """Convert a Frame dataclass to a JSON-serializable dict."""
    return {
        "tick": frame.tick,
        "time": frame.time,
        "agents": [asdict(a) for a in frame.agents],
        "links": [asdict(l) for l in frame.links],  # noqa: E741
        "particles": [asdict(p) for p in frame.particles],
        "label_regions": [asdict(r) for r in frame.label_regions],
    }


@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming frames at ~30 FPS.

    Streams Frame objects containing all visual elements for rendering.
    Send empty dict {} if no frame is available yet.
    """
    await manager.connect_frames(websocket)
    sim = get_sim_state()

    try:
        target_fps = 30.0
        interval = 1.0 / target_fps

        while True:
            start = asyncio.get_event_loop().time()

            # Get latest frame - send even if None (as empty marker)
            frame = sim.latest_frame
            if frame is not None:
                frame_data = _frame_to_dict(frame)
                await websocket.send_json(frame_data)
            else:
                # Send empty frame marker so clients know we're connected
                empty_frame = {
                    "tick": -1,
                    "time": 0.0,
                    "agents": [],
                    "links": [],
                    "particles": [],
                    "label_regions": [],
                }
                await websocket.send_json(empty_frame)

            # Sleep to maintain ~30 FPS
            elapsed = asyncio.get_event_loop().time() - start
            sleep_time = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        manager.disconnect_frames(websocket)
    except Exception as e:
        logger.error("Frame streaming error: %s", str(e))
        manager.disconnect_frames(websocket)


class ControlCommand(Enum):
    """Valid control commands."""

    PLAY = "play"
    PAUSE = "pause"
    SET_SPEED = "set_speed"
    RESET = "reset"


@app.websocket("/ws/control")
async def websocket_control(websocket: WebSocket) -> None:
    """WebSocket endpoint for receiving control commands.

    Accepts commands:
    - {"type": "play"} - Resume simulation
    - {"type": "pause"} - Pause simulation
    - {"type": "set_speed", "speed": 2.0} - Set speed multiplier
    - {"type": "reset"} - Reset world to initial state
    """
    await manager.connect_control(websocket)
    sim = get_sim_state()

    try:
        while True:
            data = await websocket.receive_json()
            cmd_type = data.get("type", "").lower()

            response: dict[str, Any] = {"success": False, "message": "Unknown command"}

            if cmd_type == "play":
                sim.paused = False
                response = {"success": True, "message": "Simulation playing"}
            elif cmd_type == "pause":
                sim.paused = True
                response = {"success": True, "message": "Simulation paused"}
            elif cmd_type == "set_speed":
                speed = data.get("speed", 1.0)
                try:
                    sim.speed = float(speed)
                    response = {"success": True, "message": f"Speed set to {sim.speed}"}
                except (TypeError, ValueError):
                    response = {"success": False, "message": "Invalid speed value"}
            elif cmd_type == "reset":
                sim.reset()
                response = {"success": True, "message": "World reset"}
            else:
                response = {"success": False, "message": f"Unknown command: {cmd_type}"}

            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect_control(websocket)
    except Exception as e:
        logger.error("Control WebSocket error: %s", str(e))
        manager.disconnect_control(websocket)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
