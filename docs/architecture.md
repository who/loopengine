# LoopEngine Architecture

This document describes the system architecture for developers working with or extending LoopEngine.

## Overview

LoopEngine is a simulation framework for modeling any agent in any system using a universal schema. The architecture follows a layered design that cleanly separates concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│              (JavaScript visualization layer)                    │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Server                                  │
│          (FastAPI: REST API + WebSocket streaming)              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌───────────────────┐ ┌─────────────┐ ┌─────────────────┐
│    Projection     │ │   Engine    │ │   Discovery     │
│  (World → Frame)  │ │ (Simulation)│ │ (AI Schemas)    │
└───────────────────┘ └─────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Model                                   │
│        (World, Agent, Link, Particle, Label, Genome)            │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Architecture

### 1. Model Layer (`src/loopengine/model/`)

The foundation containing all core data structures. Everything is represented as immutable-ish Python dataclasses.

**Key Components:**

| File | Description |
|------|-------------|
| `world.py` | Container for all simulation state: agents, links, particles, labels, schemas |
| `agent.py` | Individual entity with OODA loop, genome, policy, and internal state |
| `link.py` | Typed directional connection between agents (hierarchical, peer, service, competitive) |
| `particle.py` | Discrete unit of flow traveling along links (orders, messages, resources) |
| `label.py` | Shared context (constraints, resources, norms) for groups of agents |
| `genome.py` | AI-discovered traits and their schemas defining agent variations |

**Data Model:**

```
World
├── agents: dict[str, Agent]       # All simulation agents
├── links: dict[str, Link]         # Connections between agents
├── particles: dict[str, Particle] # Active particles in transit
├── labels: dict[str, Label]       # Shared contexts
├── schemas: dict[str, GenomeSchema] # Trait definitions per role
├── external_inputs: list[ExternalInput] # External event sources
└── tick, time, speed             # Simulation clock
```

**Agent OODA Loop:**

Each agent cycles through four phases:

```
    SENSE → ORIENT → DECIDE → ACT
      │                        │
      │    (loop_period ticks) │
      └────────────────────────┘
```

- **SENSE**: Read input_buffer into internal_state
- **ORIENT**: Interpret inputs through genome biases
- **DECIDE**: Execute policy function to plan actions
- **ACT**: Emit particles (output_buffer) to the world

### 2. Engine Layer (`src/loopengine/engine/`)

The simulation heart. Drives the world forward through discrete time steps.

**Key Components:**

| File | Description |
|------|-------------|
| `simulation.py` | Main `tick_world()` function executing one simulation tick |
| `loop.py` | Agent OODA loop stepper (`step_agent()`) |
| `ga.py` | Genetic algorithm for evolving agent genomes |
| `fitness.py` | Role-specific fitness functions for GA evaluation |
| `forces.py` | Force-directed layout engine for agent positioning |
| `flexibility.py` | Flexibility-based perturbation injection |

**Tick Sequence (simulation.py):**

```
1. Generate external inputs (roll against rate × schedule)
2. Step all agents (OODA loop)
3. Advance all particles (progress += speed)
4. Deliver particles where progress >= 1.0
5. Garbage collect dead particles
6. Update force-directed layout
7. Increment world.tick and world.time
```

**Genetic Algorithm Flow:**

```
┌─────────────────────────────────────────────────────┐
│                   GAEngine                          │
├─────────────────────────────────────────────────────┤
│  initialize_population(schema)                      │
│           │                                         │
│           ▼                                         │
│  ┌──────────────┐                                   │
│  │  Population  │◄────────────────────────────┐     │
│  └──────────────┘                             │     │
│           │                                   │     │
│           ▼                                   │     │
│  evaluate_population(fitness_fn)              │     │
│           │                                   │     │
│           ▼                                   │     │
│  select() (rank or tournament)                │     │
│           │                                   │     │
│           ▼                                   │     │
│  crossover() + mutate() → offspring           │     │
│           │                                   │     │
│           └───────────────────────────────────┘     │
│                                                     │
│  Output: best_genome, best_fitness                  │
└─────────────────────────────────────────────────────┘
```

### 3. Projection Layer (`src/loopengine/projection/`)

Transforms World state into visual frames for the frontend.

**Key Components:**

| File | Description |
|------|-------------|
| `projector.py` | `project(World) → Frame` conversion |

**Frame Structure:**

```
Frame
├── tick, time
├── agents: list[AgentVisual]     # Position, breathing, glow
├── links: list[LinkVisual]       # Bezier curves, thickness
├── particles: list[ParticleVisual] # Position along path
└── label_regions: list[LabelRegionVisual] # Convex hull clouds
```

**Visual Features:**

- **Agents**: Breathing animation (radius oscillation), glow intensity based on input buffer depth
- **Links**: Bezier curves with sway animation, thickness based on bandwidth
- **Particles**: Interpolated position along link path with color by type
- **Labels**: Translucent convex hull regions around agent groups

### 4. Server Layer (`src/loopengine/server/`)

FastAPI application providing REST API and WebSocket streaming.

**Key Components:**

| File | Description |
|------|-------------|
| `app.py` | Main FastAPI application with all endpoints |

**Endpoints:**

| Type | Endpoint | Description |
|------|----------|-------------|
| REST | `GET /api/world` | Current world state summary |
| REST | `GET /api/corpora` | List available corpora |
| REST | `GET /api/agents` | List all agents |
| REST | `GET /api/agents/{id}` | Get specific agent |
| REST | `GET /api/links` | List all links |
| REST | `GET /api/schemas` | List genome schemas |
| REST | `POST /api/world/reset` | Reset to initial state |
| REST | `POST /api/world/load_corpus` | Load a named corpus |
| REST | `POST /api/world/pause` | Pause simulation |
| REST | `POST /api/world/play` | Resume simulation |
| REST | `POST /api/world/speed` | Set speed multiplier |
| REST | `POST /api/ga/run` | Start GA evolution |
| REST | `GET /api/ga/status/{job_id}` | Get GA job status |
| REST | `POST /api/discovery/run` | Start schema discovery |
| REST | `GET /api/discovery/status/{job_id}` | Get discovery status |
| WS | `/ws/frames` | Stream Frame objects at ~30 FPS |
| WS | `/ws/control` | Send control commands (play/pause/speed/GA) |

**Thread Model:**

```
┌────────────────────────────────────────────────────┐
│                  Main Thread                        │
│  (FastAPI async event loop)                        │
├────────────────────────────────────────────────────┤
│  - REST endpoint handlers                          │
│  - WebSocket connection management                 │
│  - Frame broadcasting                              │
└────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│  Simulation     │  │  GA Worker      │
│  Thread         │  │  Pool           │
├─────────────────┤  ├─────────────────┤
│  tick_world()   │  │  Evolution jobs │
│  at ~30 ticks/s │  │  (ThreadPool)   │
└─────────────────┘  └─────────────────┘
```

### 5. Discovery Layer (`src/loopengine/discovery/`)

AI-powered genome schema discovery using Claude API.

**Key Components:**

| File | Description |
|------|-------------|
| `discoverer.py` | `Discoverer` class for schema discovery |
| `triggers.py` | Conditions that trigger rediscovery |

**Discovery Flow:**

```
System Description (roles, inputs, outputs, constraints)
                    │
                    ▼
           ┌───────────────┐
           │   Discoverer  │
           │  (Claude API) │
           └───────────────┘
                    │
                    ▼
           DiscoveryResult
           ├── roles: dict[str, DiscoveredRole]
           │   ├── schema: GenomeSchema
           │   └── flexibility_score: float
           └── discovery_prompt
                    │
                    ▼
           migrate_genome(old_genome, new_schema)
           ├── Added traits (random init)
           ├── Preserved traits (kept)
           └── Vestigial traits (marked)
```

### 6. Behaviors Layer (`src/loopengine/behaviors/`)

AI-driven behavior generation for agents (optional LLM integration).

**Key Components:**

| File | Description |
|------|-------------|
| `ai_behavior_engine.py` | Main orchestrator for behavior generation |
| `llm_client.py` | Abstract LLM client interface |
| `prompt_builder.py` | Builds prompts from domain/agent context |
| `response_parser.py` | Parses LLM responses into actions |
| `fallback.py` | Fallback behaviors when LLM unavailable |
| `rate_limiter.py` | Rate limit handling with retries |
| `providers/` | Provider-specific clients (Claude, OpenAI, Ollama) |

### 7. Corpora Layer (`src/loopengine/corpora/`)

Pre-built simulation scenarios demonstrating the framework.

**Available Corpora:**

| Corpus | Description |
|--------|-------------|
| `sandwich_shop/` | Small shop with owner, sandwich maker, cashier |
| `software_team/` | Software team with PM, developers, designer |

**Corpus Structure:**

Each corpus provides a `create_world()` function that returns a fully configured World with:

- Agents (with genomes and policies)
- Links (relationship types and properties)
- Labels (shared contexts)
- External inputs (event sources)

## Data Flow

### World → Frame → Canvas

```
┌─────────────┐     tick_world()      ┌─────────────┐
│    World    │──────────────────────▶│    World    │
│  (tick N)   │                       │  (tick N+1) │
└─────────────┘                       └─────────────┘
                                              │
                                              │ project()
                                              ▼
                                      ┌─────────────┐
                                      │    Frame    │
                                      │ (visuals)   │
                                      └─────────────┘
                                              │
                                              │ WebSocket
                                              ▼
                                      ┌─────────────┐
                                      │   Canvas    │
                                      │ (frontend)  │
                                      └─────────────┘
```

### Particle Flow

```
External Input                    Agent A                     Agent B
      │                              │                            │
      │ spawn_particle()             │                            │
      ├─────────────────────────────▶│                            │
      │                              │ policy() → Particle        │
      │                              ├───────────────────────────▶│
      │                              │     (travels on link)      │
      │                              │                            │
      │                              │        progress += speed   │
      │                              │                            │
      │                              │     when progress >= 1.0   │
      │                              │                            │
      │                              │        deliver to          │
      │                              │        input_buffer        │
      │                              │                            ▼
```

### GA Evaluation Flow

```
GAEngine                FitnessEvaluator              World Clone
    │                          │                           │
    │ evaluate_fitness(genome) │                           │
    ├─────────────────────────▶│                           │
    │                          │ clone_world()             │
    │                          ├──────────────────────────▶│
    │                          │                           │
    │                          │ apply genome to agent     │
    │                          ├──────────────────────────▶│
    │                          │                           │
    │                          │ run N ticks               │
    │                          ├──────────────────────────▶│
    │                          │                           │
    │                          │ compute fitness_fn(world) │
    │                          │◀──────────────────────────┤
    │                          │                           │
    │◀─────────────────────────┤                           │
    │       fitness score      │                           │
```

## Extension Points

### Adding a New Corpus

1. Create directory `src/loopengine/corpora/your_corpus/`
2. Create `__init__.py` with `create_world() -> World`
3. Define agents with roles, genomes, and policy functions
4. Define links with appropriate types
5. Define labels for shared contexts
6. Define external inputs for event generation
7. Register in `SimulationState.AVAILABLE_CORPORA` in `server/app.py`

### Adding a New Agent Role

1. Define genome traits relevant to the role
2. Create a policy function: `policy(sensed_inputs, genome, internal_state) -> list[Particle]`
3. Add fitness function if GA evolution is needed
4. Create Agent instance with appropriate loop_period and labels

### Adding a New Link Type

1. Add to `LinkType` enum in `model/link.py`
2. Update `_compute_hierarchical_forces()` in `forces.py` if special layout behavior needed
3. Add color mapping in `LINK_COLORS` in `projection/projector.py`

### Adding a New Particle Type

1. Use any string for `particle_type`
2. Add color mapping in `PARTICLE_COLORS` in `projection/projector.py`
3. Handle in agent policies as needed

### Adding a New Force Type

1. Add computation function `_compute_your_forces()` in `forces.py`
2. Call from `compute_forces()`
3. Add configuration parameters to `ForceConfig` if needed

## Design Decisions

### Why Dataclasses?

- Simple, readable data structures
- Type hints built-in
- Easy serialization
- Mutable when needed (simulation state)

### Why OODA Loop?

- Universal model for any decision-making agent
- Clearly separates sensing, processing, and acting
- Configurable timing via loop_period
- Supports both synchronous and asynchronous processing

### Why Force-Directed Layout?

- Organic, emergent positioning
- Automatically reveals structure (hierarchy, clusters)
- Responds to link density and relationships
- Continuous animation possibilities

### Why Separate Projection Layer?

- Clean separation between simulation and visualization
- Frontend-agnostic (can swap visualization)
- Frame contains only what's needed for rendering
- Enables performance optimization (limit particles rendered)

### Why Background GA Jobs?

- Evolution is computationally expensive
- Non-blocking UI experience
- Progress tracking via WebSocket broadcasts
- Cancellation support

## Performance Considerations

- **Particle limits**: MAX_PARTICLES=100 prevents memory growth
- **Repulsion cutoff**: Skip force calculations for distant agents (500px)
- **Rendered particle limit**: MAX_RENDERED_PARTICLES=100 for frontend performance
- **Concurrent behavior generation**: Thread pool for 50+ agents

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for discovery/behaviors | - |
| `BEHAVIOR_CACHE_TTL` | Cache TTL for generated behaviors | 300s |
| `LOG_LEVEL` | Logging level | INFO |
