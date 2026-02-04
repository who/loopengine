# Product Requirements Document: LoopEngine

## Universal Agent Ontology Simulator

**Version:** 0.1.0-alpha
**Date:** February 2026
**Status:** Draft
**First Implementation Corpus:** The Sandwich Shop (3 Agents)

---

## 1. Vision

LoopEngine is a simulation framework and visual explorer for modeling any agent in any system using a universal schema. Every agent — a line cook, a CEO, a janitor, a politician — is described by the same formal structure: inputs, outputs, a genome of capabilities, a behavioral policy, typed links to other agents, shared labels, and a looping sense-orient-decide-act cycle. The framework evolves agent genomes using genetic algorithms, uses AI to autonomously discover genome properties, and renders the living system as an organic, breathing canvas visualization.

### 1.1 First Principle

**Everything is a loop.** Every agent runs a loop. Every organization emerges from interlocking loops. The genetic algorithm is a loop. The AI genome discovery process is a loop. Loops nest within loops at different frequencies. The framework makes these loops visible and evolvable.

### 1.2 Core Assertions

- **Agents are always individuals.** Never teams, never groups, never departments. A team is a cluster of individual agents who share a label and proximity.
- **Labels are shared context, not actors.** A label like "Kitchen Staff" describes which agents share constraints, norms, and resources. It does not act.
- **Links are typed, directional, and carry properties.** A hierarchical link from a supervisor to a worker carries authority scope, fitness definition, autonomy granted, and directional flow channels. A peer link between coworkers carries coordination and information exchange properties.
- **Genomes are discovered, not hardcoded.** AI examines the system description and infers what dimensions of variation are meaningful for each role. Genome schemas are periodically rediscovered as systems evolve.
- **The visualization breathes.** Agents pulse, particles flow, conduits sway. The system's health is legible at a glance from rhythm, density, and flow patterns — before reading a single label.

---

## 2. Architecture Overview

The system is split into three clean layers. No layer knows about the internals of another. Communication flows through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                    │
│                                                         │
│   Canvas Renderer ← Frame Snapshots ← WebSocket/API    │
│   User Interaction → Commands → WebSocket/API           │
└─────────────────────────────┬───────────────────────────┘
                              │ HTTP / WebSocket
┌─────────────────────────────┴───────────────────────────┐
│                    BACKEND (Python / uv)                 │
│                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│   │  Model Layer  │  │  Loop Engine  │  │  Frame        │ │
│   │  (Domain Data)│→ │  (Simulation) │→ │  Projector    │ │
│   └──────────────┘  └──────┬───────┘  └──────────────┘ │
│                            │                             │
│   ┌──────────────┐  ┌──────┴───────┐                    │
│   │  AI Discovery │  │  GA Engine   │                    │
│   │  (LLM-driven) │→ │  (Evolution) │                    │
│   └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### 2.1 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Package/project management** | `uv` | Fast, modern Python project management. Handles dependencies, virtual environments, and scripts. |
| **Python version** | 3.12+ | f-strings, match statements, dataclasses, type hints throughout. |
| **Backend web framework** | FastAPI | Async-native, WebSocket support, automatic OpenAPI docs. |
| **WebSocket** | FastAPI WebSockets + `uvicorn` | Real-time frame streaming to the frontend. |
| **Frontend** | Vanilla HTML5 Canvas + JavaScript | No framework. The rendering is custom and particle-based. Keeping it dependency-free. |
| **AI Discovery** | Anthropic Claude API | System descriptions sent as prompts; genome schemas returned as structured JSON. |
| **Data serialization** | `msgpack` or JSON | Frame snapshots streamed as compact payloads. |
| **Testing** | `pytest` | Standard Python test framework. |
| **Linting/formatting** | `ruff` | Fast, uv-compatible linter and formatter. |

### 2.2 Project Structure

```
loopengine/
├── pyproject.toml                 # uv project definition
├── uv.lock                        # locked dependencies
├── README.md
├── src/
│   └── loopengine/
│       ├── __init__.py
│       ├── model/                 # Layer 1: Domain model
│       │   ├── __init__.py
│       │   ├── agent.py           # Agent dataclass
│       │   ├── link.py            # Link dataclass
│       │   ├── particle.py        # Particle dataclass
│       │   ├── label.py           # Label + Context dataclass
│       │   ├── genome.py          # GenomeSchema + Genome
│       │   └── world.py           # World container
│       ├── engine/                # Layer 2: Simulation loops
│       │   ├── __init__.py
│       │   ├── loop.py            # Agent OODA loop stepper
│       │   ├── simulation.py      # World-level tick driver
│       │   ├── ga.py              # Genetic algorithm loop
│       │   └── forces.py          # Force-directed layout
│       ├── discovery/             # AI genome discovery
│       │   ├── __init__.py
│       │   └── discoverer.py      # LLM-based schema inference
│       ├── projection/            # Layer 3: Frame generation
│       │   ├── __init__.py
│       │   └── projector.py       # World → Frame snapshot
│       ├── server/                # API + WebSocket server
│       │   ├── __init__.py
│       │   └── app.py             # FastAPI application
│       └── corpora/               # Test corpora / scenarios
│           ├── __init__.py
│           └── sandwich_shop.py   # The first corpus
├── frontend/
│   ├── index.html                 # Single HTML entry point
│   ├── style.css                  # Minimal styling
│   └── js/
│       ├── main.js                # Entry point, WebSocket connection
│       ├── renderer.js            # Canvas drawing loop
│       ├── particles.js           # Particle rendering + trails
│       ├── agents.js              # Agent shape rendering + breathing
│       ├── links.js               # Conduit rendering + sway
│       ├── labels.js              # Label region cloud rendering
│       └── interaction.js         # Hover, click, zoom, pan
└── tests/
    ├── test_model.py
    ├── test_engine.py
    ├── test_projection.py
    └── test_sandwich_shop.py
```

---

## 3. Backend: Model Layer

### 3.1 Agent

```python
@dataclass
class Agent:
    id: str
    name: str
    role: str
    genome: dict[str, float]       # trait_name → value (0.0 to 1.0)
    labels: set[str]               # e.g. {"SandwichShop", "Kitchen"}
    internal_state: dict[str, Any] # arbitrary role-specific state
    input_buffer: list[Particle]
    output_buffer: list[Particle]

    # Loop configuration
    loop_period: int               # ticks per full OODA revolution
    loop_phase: Phase              # SENSE | ORIENT | DECIDE | ACT
    phase_tick: int                # current tick within phase

    # Policy: callable(sensed_inputs, genome, internal_state) → list[Particle]
    policy: Callable

    # Position (for force-directed layout)
    x: float
    y: float
    vx: float                      # velocity for physics integration
    vy: float
```

The `Phase` is an enum: `SENSE`, `ORIENT`, `DECIDE`, `ACT`. Each phase occupies `loop_period / 4` ticks. When `phase_tick` reaches the phase duration, the agent advances to the next phase. When ACT completes, the loop wraps back to SENSE.

The `genome` is a plain `dict[str, float]`. Trait names are strings. Values are normalized floats between 0.0 and 1.0. The dictionary is not constrained to any fixed set of keys at the code level — the GenomeSchema provides the expected keys, but the agent doesn't enforce conformance. This keeps things flexible for AI-discovered schema changes.

The `policy` callable is the agent's brain. For the initial implementation, policies are hand-written functions specific to each role. Later, the GA can evolve policies or the parameters that govern them.

### 3.2 Link

```python
@dataclass
class Link:
    id: str
    source_id: str                 # agent id
    dest_id: str                   # agent id
    link_type: LinkType            # HIERARCHICAL | PEER | SERVICE | COMPETITIVE
    properties: dict[str, Any]
    # Properties may include:
    #   authority_scope: list[str]
    #   autonomy_granted: float
    #   fitness_definition: list[str]
    #   resource_control: list[str]
    #   flow_types: list[str]      # what particle types travel this link
    #   bandwidth: float           # how many particles per tick can traverse
    #   latency: int               # ticks for a particle to traverse
```

`LinkType` is an enum: `HIERARCHICAL`, `PEER`, `SERVICE`, `COMPETITIVE`.

Links are directional. A bidirectional relationship (like Alex ↔ Tom passing tickets and sandwiches) is modeled as two separate Link objects, potentially with different properties. The ticket link from Alex → Tom may have different bandwidth than the sandwich link from Tom → Alex.

### 3.3 Particle

```python
@dataclass
class Particle:
    id: str
    particle_type: str             # "order_ticket", "sandwich", "directive", etc.
    payload: dict[str, Any]        # arbitrary data carried by the particle
    source_id: str                 # originating agent id
    dest_id: str                   # destination agent id
    link_id: str                   # which link it's traveling on
    progress: float                # 0.0 (at source) to 1.0 (at destination)
    speed: float                   # progress increment per tick
    alive: bool                    # set to False when delivered or expired
```

When `progress >= 1.0`, the particle is delivered: it gets appended to the destination agent's `input_buffer` and its `alive` flag is set to `False`. Dead particles are garbage-collected at the end of each tick.

### 3.4 Label

```python
@dataclass
class Label:
    name: str                      # e.g. "SandwichShop_Main"
    context: LabelContext

@dataclass
class LabelContext:
    constraints: list[str]         # e.g. ["health_code", "operating_hours_9_to_9"]
    resources: list[str]           # e.g. ["ingredient_supply_chain", "POS_system"]
    norms: list[str]               # e.g. ["FIFO_inventory", "customer_greeting"]
    description: str               # human-readable description
```

Labels don't reference agents. Agents carry label names in their `labels` set. To find all agents with a given label, query the World's agent collection.

### 3.5 GenomeSchema

```python
@dataclass
class GenomeTrait:
    name: str
    description: str
    min_val: float                 # typically 0.0
    max_val: float                 # typically 1.0
    category: str                  # e.g. "physical", "cognitive", "social"
    discovered_at: datetime        # when the AI discovered this trait

@dataclass
class GenomeSchema:
    role: str                      # e.g. "sandwich_maker"
    traits: dict[str, GenomeTrait]
    discovered_at: datetime
    discovery_prompt: str          # the system description sent to the AI
    version: int
```

The GenomeSchema is the AI's output. It defines what traits are meaningful for a given role. Individual agent genomes are instances of this schema, but loosely coupled — genomes may have extra or missing keys relative to the current schema.

### 3.6 World

```python
@dataclass
class World:
    agents: dict[str, Agent]       # id → Agent
    links: dict[str, Link]         # id → Link
    particles: dict[str, Particle] # id → Particle (active only)
    labels: dict[str, Label]       # name → Label
    schemas: dict[str, GenomeSchema]  # role → GenomeSchema

    tick: int                      # global tick counter
    time: float                    # elapsed simulation time
    speed: float                   # ticks per second (simulation speed)

    # External input configuration
    external_inputs: list[ExternalInput]  # e.g. customer arrival process

@dataclass
class ExternalInput:
    name: str                      # e.g. "customer_arrivals"
    target_agent_id: str           # who receives the input
    rate: float                    # average particles per tick
    variance: float                # randomness in arrival rate
    particle_type: str             # what kind of particle is generated
    payload_generator: Callable    # function that creates particle payloads
    schedule: Callable             # function(tick) → rate_multiplier (for rush patterns)
```

---

## 4. Backend: Loop Engine

### 4.1 Agent Loop (Innermost)

Each tick, every agent advances one step in its OODA loop:

```
SENSE phase:
    Read input_buffer → store as sensed_inputs in internal_state
    Clear input_buffer
    Genome traits like "observation" and "signal_discrimination" can filter
    or weight the sensed inputs

ORIENT phase:
    Interpret sensed_inputs through genome biases
    E.g., a risk-tolerant agent interprets ambiguous signals as opportunity
    Update internal_state with oriented interpretation

DECIDE phase:
    Run policy(sensed_inputs, genome, internal_state) → planned_actions
    Store planned_actions in internal_state

ACT phase:
    Convert planned_actions into Particle objects
    Place particles on appropriate outgoing links (into world.particles)
    Update any internal_state side effects

Phase advance:
    Increment phase_tick
    If phase_tick >= (loop_period / 4):
        Advance to next phase, reset phase_tick to 0
        If wrapping from ACT → SENSE: one full revolution complete
```

The loop stepper is a pure function: `step_agent(agent: Agent, world: World) → list[Particle]`. It returns newly created particles. It mutates `agent.internal_state`, `agent.input_buffer`, `agent.loop_phase`, and `agent.phase_tick`. It does not mutate anything else in the world.

### 4.2 Simulation Loop (Per Tick)

One tick of the simulation:

```
1. Generate external inputs
   For each ExternalInput, roll against rate * schedule(tick) to decide
   whether to spawn a particle. If so, create it and place it on the
   appropriate link or directly into the target agent's input buffer.

2. Step all agents
   For each agent in world.agents, call step_agent.
   Collect all newly emitted particles.

3. Advance all particles
   For each particle in world.particles where alive is True:
       particle.progress += particle.speed
       If progress >= 1.0:
           Deliver to destination agent's input_buffer
           particle.alive = False

4. Garbage collect dead particles
   Remove all particles where alive is False from world.particles.

5. Update force-directed layout
   Compute forces on all agents (spring attraction along links,
   repulsion between agents, vertical bias for hierarchy, label cohesion).
   Integrate forces into agent velocities and positions.
   Apply damping to prevent oscillation.

6. Advance world clock
   world.tick += 1
   world.time += (1.0 / world.speed)
```

### 4.3 Genetic Algorithm Loop

The GA operates on a population of agents for a given role:

```
INITIALIZE:
    Given a GenomeSchema for a role, generate N agents with random
    genome values (uniform within each trait's range).

EVALUATE (one generation):
    For each candidate genome:
        Create an agent with that genome.
        Place it into a copy of the world (with fixed other agents).
        Run the simulation for T ticks.
        Compute fitness based on the role's fitness function.
    Sort candidates by fitness.

SELECT:
    Top K candidates survive (tournament or rank selection).

CROSSOVER:
    Pair surviving candidates. For each pair, produce offspring by
    combining traits. Strategies:
        - Uniform crossover: each trait independently chosen from one parent.
        - Blended crossover: each trait is a weighted average of parents.

MUTATE:
    For each offspring, with probability P_mut per trait:
        Perturb the trait value by a small random amount (Gaussian noise).
        Clamp to [min_val, max_val].

REPEAT:
    New population = survivors + offspring.
    Run EVALUATE again. Continue for G generations.

OUTPUT:
    Best genome found. Optionally, the full Pareto frontier if
    fitness is multi-objective.
```

**Fitness functions** are role-specific callables: `fitness(agent: Agent, world: World, history: list[WorldSnapshot]) → float`. For the sandwich shop:

- **Tom's fitness:** sandwiches produced per tick × consistency score − waste generated − queue backup penalty.
- **Alex's fitness:** customers served per tick × order accuracy − customer wait time.
- **Maria's fitness:** aggregate shop throughput × (1 − supply_cost_ratio) − stockout_count − waste.

### 4.4 Force-Directed Layout

The layout engine runs each tick (or every N ticks for performance). It computes four forces per agent:

**Link spring force:** For each link involving the agent, apply a spring force pulling toward the linked agent. Spring strength is proportional to the link's interaction density (how many particles have recently traversed it). Rest length is a configurable constant.

**Agent repulsion:** For each other agent, apply a gentle inverse-square repulsion to prevent overlap.

**Hierarchical vertical bias:** For hierarchical links, apply a gentle upward force on the authority end and downward on the subordinate end. This produces a natural vertical stratification without forcing a rigid tree layout.

**Label cohesion:** For each label the agent carries, apply a gentle force toward the centroid of all agents sharing that label. This pulls label groups into loose clusters.

Forces are summed per agent, integrated into velocity (Euler or Verlet integration), damped, and applied to position. The result is a soft, organic layout that drifts and settles based on the system's structure.

---

## 5. Backend: AI Genome Discovery

### 5.1 Discovery Process

The discoverer takes a system description (roles, links, label contexts, input/output types, constraints) and produces a GenomeSchema for each role.

```
INPUT:
    System description as structured text or JSON.
    Example for the sandwich shop:
    {
        "system": "Small sandwich shop, 3 employees",
        "roles": [
            {
                "name": "owner",
                "inputs": ["supply_invoices", "revenue_reports", "staff_status",
                           "customer_flow_observation"],
                "outputs": ["supply_orders", "directives", "schedule_changes",
                            "pricing_decisions"],
                "constraints": ["budget", "health_code", "operating_hours"],
                "links_to": ["sandwich_maker (hierarchical)", "cashier (hierarchical)"]
            },
            ...
        ]
    }

PROCESS:
    Send system description to Claude API with a prompt that asks:
    "Given this system and these roles, what are the meaningful
    dimensions along which an agent in each role could vary that would
    affect their performance? Return a GenomeSchema as JSON with
    trait names, descriptions, categories, and suggested ranges."

OUTPUT:
    GenomeSchema per role, parsed from the LLM response.
```

### 5.2 Periodic Rediscovery

The discovery loop runs on a configurable schedule. Triggers include:

- Manual invocation by the user.
- System description changes (a role is added, links are restructured).
- GA stagnation (fitness plateau across generations, suggesting the genome dimensions may be inadequate).
- Scheduled interval (e.g., every 100 GA generations).

When rediscovery runs, the new schema is compared with the existing one. New traits are added to existing agent genomes with random initialization. Deprecated traits are flagged but not immediately removed (they become vestigial — present in the genome but not referenced by the current schema). This allows graceful migration without breaking running simulations.

### 5.3 Input Randomization

The discovery process also infers a **flexibility parameter** for each role: how much input variance the agent should expect. This is a float from 0.0 (completely predictable inputs) to 1.0 (wildly unpredictable inputs).

The flexibility parameter controls:

- **During GA evolution:** How much the external input distributions are randomized across evaluation runs. A low-flexibility role (line cook) is always evaluated against the same standard input stream. A high-flexibility role (startup CEO) is evaluated against highly varied input streams, selecting for robustness.
- **During live simulation:** How much random perturbation is injected into the agent's input stream. Occasional surprise inputs test the agent's adaptability.

---

## 6. Backend: Frame Projection

### 6.1 Frame Data Structures

```python
@dataclass
class AgentVisual:
    id: str
    x: float
    y: float
    radius: float                  # base size, modulated by breathing
    breathing_phase: float         # 0 to 2π, oscillates over time
    breathing_rate: float          # radians per tick (faster = more stressed)
    color: str                     # hex color, determined by role
    glow_intensity: float          # 0.0 to 1.0, encoding load/stress
    ooda_phase: str                # "sense", "orient", "decide", "act"
    genome_summary: dict[str, float]  # for hover display
    name: str
    role: str

@dataclass
class LinkVisual:
    id: str
    control_points: list[tuple[float, float]]  # bezier/spline path
    thickness: float               # encoding interaction density
    sway_phase: float              # gentle oscillation offset
    color: str
    link_type: str

@dataclass
class ParticleVisual:
    id: str
    x: float
    y: float
    color: str
    size: float
    particle_type: str
    trail: list[tuple[float, float]]  # recent positions for fading tail
    opacity: float

@dataclass
class LabelRegionVisual:
    name: str
    hull_points: list[tuple[float, float]]  # convex hull, expanded
    fill_color: str                # with low alpha
    breathing_phase: float

@dataclass
class Frame:
    tick: int
    time: float
    agents: list[AgentVisual]
    links: list[LinkVisual]
    particles: list[ParticleVisual]
    label_regions: list[LabelRegionVisual]
```

### 6.2 Projection Logic

The projector is a pure function: `project(world: World) → Frame`.

It reads the world state and computes all visual properties:

- Agent radius = base_radius × (1 + 0.05 × sin(breathing_phase)). Breathing phase advances each tick by `breathing_rate`.
- Breathing rate is computed from input buffer size and loop frequency: agents with large input buffers or fast loops breathe faster.
- Glow intensity is proportional to input buffer occupancy (backed-up agent glows brighter).
- Link control points are computed from source and destination agent positions with a slight perpendicular offset for curvature and a sin(sway_phase) displacement.
- Link thickness is proportional to the number of particles that traversed the link in the last N ticks (rolling window).
- Particle x/y is interpolated along the link's spline path at the particle's progress value.
- Particle trail is a list of the last 5 positions.
- Label region hull is computed by finding all agents carrying the label, computing their convex hull, and expanding it outward by a padding constant. A smoothing pass rounds the corners.

### 6.3 Streaming

The server streams Frame objects to the frontend over WebSocket at a configurable framerate (default: 30 FPS). Frames are serialized as JSON (or msgpack for performance). The simulation may run at a different tick rate than the display framerate — the projector samples the latest world state at display time.

---

## 7. Frontend: Canvas Renderer

### 7.1 Overview

The frontend is a single HTML page with a full-viewport `<canvas>` element. All rendering is done in JavaScript using the Canvas 2D API. No frameworks, no dependencies. The frontend connects to the backend via WebSocket, receives Frame objects, and draws them.

The rendering loop runs via `requestAnimationFrame`. Each frame:

1. Receive latest Frame from WebSocket (or interpolate between recent frames for smoothness).
2. Clear the canvas.
3. Draw label regions (lowest layer — soft translucent clouds).
4. Draw links (conduits with sway).
5. Draw particles (small colored dots with trails).
6. Draw agents (amoeba shapes with breathing pulse and inner glow).
7. Draw hover/selection overlays if applicable.

### 7.2 Agent Rendering

Each agent is drawn as a soft, slightly irregular ellipse. The irregularity comes from a Perlin noise offset applied to the radius at multiple angles around the shape, giving it an organic, amoeba-like quality. The radius modulates with the breathing phase (gentle sinusoidal expansion and contraction).

The interior has a radial gradient — brighter at the center, fading to the membrane color at the edge. Glow intensity modulates the brightness. The OODA phase can be subtly indicated by a small inner marker or a tint shift (optional, off by default to keep the visual clean).

The agent's name is drawn in small text below the shape, only visible at sufficient zoom level.

### 7.3 Link Rendering

Links are drawn as quadratic or cubic bezier curves between agent centers. The control points are offset perpendicular to the direct line between agents, with a sinusoidal sway applied over time. Line width encodes interaction density. Color encodes link type (hierarchical links are one hue, peer links another, service links a third).

For bidirectional relationships (two separate links between the same agents), the two curves are offset to opposite sides so they form a visible pair, like a two-lane road.

### 7.4 Particle Rendering

Particles are small circles (radius 2-5 pixels depending on type) drawn at their interpolated position along the link path. Each particle has a short fading trail — the last 3-5 positions drawn with decreasing opacity, giving a sense of motion.

Particle color encodes type:
- Order tickets: warm amber
- Finished sandwiches: green
- Directives: cool blue
- Status reports: light gray
- Supply orders: orange
- Revenue/money: gold

### 7.5 Label Region Rendering

Label regions are drawn as filled shapes with very low opacity (alpha 0.08-0.15). The hull is smoothed into a rounded shape using cardinal spline interpolation. The region breathes — expanding and contracting gently. Where multiple label regions overlap, the colors blend additively, producing a richer tint.

### 7.6 Interaction

**Hover:** When the mouse is over an agent, the agent highlights (brighter glow, slightly larger). A tooltip appears showing the agent's name, role, genome traits as a small bar chart, current OODA phase, and input buffer depth. Nearby links brighten and particle types become labeled.

**Click:** Clicking an agent selects it. The canvas gently pans to center the agent. A detail panel slides in from the side showing full agent information: all genome traits, all labels, all connected links with their properties, current internal state, fitness history.

**Zoom:** Mouse wheel or pinch to zoom. At macro zoom, individual agents are dots and the overall topology is visible. At micro zoom, agent shapes, particles, labels, and genome details resolve.

**Pan:** Click and drag on empty space to pan the viewport.

**Time control:** A minimal UI bar at the bottom with play/pause, speed slider (0.25x to 10x), and a tick counter.

---

## 8. Backend: Server API

### 8.1 WebSocket Endpoints

**`ws://localhost:8000/ws/frames`**
Streams Frame objects to the frontend at the configured display rate. The frontend connects once and receives a continuous stream.

Message format (server → client):
```json
{
    "type": "frame",
    "data": { ... Frame as JSON ... }
}
```

**`ws://localhost:8000/ws/control`**
Receives commands from the frontend.

Message format (client → server):
```json
{"type": "play"}
{"type": "pause"}
{"type": "set_speed", "speed": 2.0}
{"type": "select_agent", "agent_id": "tom"}
{"type": "get_agent_detail", "agent_id": "tom"}
{"type": "trigger_discovery"}
{"type": "start_ga", "role": "sandwich_maker", "generations": 100}
```

### 8.2 REST Endpoints

**`GET /api/world`** — Returns the full World state as JSON. Useful for debugging and inspection.

**`GET /api/agents`** — Returns all agents with their current state.

**`GET /api/agents/{agent_id}`** — Returns detailed info for one agent.

**`GET /api/schemas`** — Returns all current GenomeSchemas.

**`POST /api/discovery/run`** — Triggers AI genome discovery. Accepts a system description in the body. Returns the new schemas.

**`POST /api/ga/run`** — Triggers a GA run for a given role. Accepts role name and generation count. Returns best genome found.

**`POST /api/world/reset`** — Resets the simulation to initial state.

**`POST /api/world/load_corpus`** — Loads a named corpus (e.g., "sandwich_shop"). Initializes the world with that scenario.

---

## 9. First Corpus: The Sandwich Shop

### 9.1 System Description

A small sandwich shop with three employees. Customers arrive, place orders at the register, sandwiches are assembled and delivered. The owner manages supplies and oversees operations.

### 9.2 Agents

**Maria — Owner**
- **Role:** `owner`
- **Labels:** `{"SandwichShop", "Management"}`
- **Loop period:** 300 ticks (slow loop — she observes and adjusts over long intervals)
- **Inputs:** Revenue reports from Alex, status signals from Tom and Alex, supply level observations, customer flow observations
- **Outputs:** Supply orders (to external), directives (to Tom and Alex), schedule adjustments
- **Initial genome (before AI discovery):**
  - `supply_forecasting: 0.7`
  - `observation: 0.8`
  - `decisiveness: 0.6`
  - `delegation: 0.7`
  - `cost_sensitivity: 0.9`
- **Policy:** On each SENSE, observe shop state (queue depth, supply levels, throughput rate). On DECIDE, if queue depth exceeds threshold, send directive to Alex to speed up or to Tom to simplify. If supply levels low, generate a supply order particle. If throughput is healthy, do nothing (conserve attention).

**Tom — Sandwich Maker**
- **Role:** `sandwich_maker`
- **Labels:** `{"SandwichShop", "FrontLine", "Kitchen"}`
- **Loop period:** 30 ticks (fast loop — one sandwich cycle)
- **Inputs:** Order tickets from Alex, ingredients from supply, directives from Maria
- **Outputs:** Finished sandwiches to Alex, status reports to Maria, waste (to external)
- **Initial genome:**
  - `speed: 0.7`
  - `consistency: 0.8`
  - `ingredient_intuition: 0.6`
  - `stress_tolerance: 0.7`
  - `waste_minimization: 0.5`
- **Policy:** On SENSE, read next ticket from input buffer. On ORIENT, check ingredient availability for the order. On DECIDE, if ingredients available, plan assembly; if not, substitute (governed by `ingredient_intuition`) or report stockout to Maria. On ACT, produce a sandwich particle (quality modulated by `consistency` and `speed` — faster agents may sacrifice consistency) and send to Alex.

**Alex — Cashier**
- **Role:** `cashier`
- **Labels:** `{"SandwichShop", "FrontLine", "Register"}`
- **Loop period:** 20 ticks (fastest loop — rapid customer interactions)
- **Inputs:** Customer particles from external, finished sandwiches from Tom, directives from Maria
- **Outputs:** Order tickets to Tom, served customers to external, revenue reports to Maria
- **Policy:** On SENSE, check for waiting customer in input buffer. On ORIENT, read customer order. On DECIDE, create order ticket. On ACT, emit ticket particle to Tom, process payment. When a sandwich particle arrives from Tom, match it to the waiting customer and emit a served_customer particle.

### 9.3 Links

| Link ID | Source | Dest | Type | Key Properties |
|---|---|---|---|---|
| `maria_to_tom` | Maria | Tom | HIERARCHICAL | authority_scope: [recipe_standards, supply_priorities], autonomy_granted: 0.5, fitness_definition: [speed, consistency, waste], flow_types: [directive] |
| `tom_to_maria` | Tom | Maria | HIERARCHICAL (upward) | flow_types: [status_report, stockout_alert] |
| `maria_to_alex` | Maria | Alex | HIERARCHICAL | authority_scope: [service_standards, upselling_policy], autonomy_granted: 0.4, fitness_definition: [throughput, accuracy, friendliness], flow_types: [directive] |
| `alex_to_maria` | Alex | Maria | HIERARCHICAL (upward) | flow_types: [revenue_report, status_report] |
| `alex_to_tom` | Alex | Tom | SERVICE | flow_types: [order_ticket], bandwidth: 1.0 |
| `tom_to_alex` | Tom | Alex | SERVICE | flow_types: [finished_sandwich], bandwidth: 1.0 |

### 9.4 Labels

| Label | Context |
|---|---|
| `SandwichShop` | constraints: [health_code, operating_hours_10_to_8, max_capacity_30], resources: [ingredient_supply, POS_system, kitchen_equipment], norms: [FIFO_orders, customer_greeting, clean_as_you_go] |
| `FrontLine` | constraints: [customer_facing_appearance], norms: [friendly_demeanor, no_phone_use] |
| `Kitchen` | constraints: [food_safety_gloves, temperature_monitoring], resources: [grill, prep_station, cold_storage] |
| `Management` | resources: [supplier_contacts, POS_admin, scheduling_system], norms: [daily_inventory_check, weekly_supply_order] |
| `Register` | resources: [POS_terminal, cash_drawer], constraints: [cash_handling_policy] |

### 9.5 External Inputs

**Customer arrivals:**
- Target: Alex
- Base rate: 0.05 particles per tick (roughly 1 customer every 20 ticks)
- Schedule: Multiplier function that models a lunch rush — rate doubles between ticks 200-400, returns to baseline outside that window
- Variance: 0.3 (moderate randomness around the rate)
- Particle type: `customer_order`
- Payload: randomly generated sandwich order (type, extras, special_requests)

**Supply deliveries:**
- Target: Tom (via Maria's supply ordering)
- Modeled as a response to Maria's supply_order particles with a fixed latency (e.g., 50 ticks after the order is placed, an ingredient_delivery particle appears in Tom's input buffer)

### 9.6 Fitness Functions

**Tom:**
```
fitness = (sandwiches_completed / ticks_elapsed)        # throughput
         × average_consistency_score                     # quality
         − (waste_particles / total_ingredients_used)    # efficiency
         − (max_queue_depth × 0.1)                       # responsiveness penalty
```

**Alex:**
```
fitness = (customers_served / ticks_elapsed)             # throughput
         × order_accuracy_rate                            # quality
         − (average_customer_wait_ticks × 0.05)           # speed penalty
         + (upsell_count × 0.1)                           # bonus revenue
```

**Maria:**
```
fitness = shop_total_throughput                           # overall productivity
         × (1 - supply_cost / revenue)                   # margin
         − stockout_events × 2.0                          # severe penalty for running out
         − waste_total × 0.5                              # waste penalty
```

### 9.7 Simulation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Ticks per simulation run | 1000 | Enough for ~3 full Maria loops, ~33 Tom loops, ~50 Alex loops |
| GA population size | 50 | Small enough for fast iteration in v0.1 |
| GA generations | 100 | Sufficient for convergence on a small genome |
| Mutation rate | 0.1 per trait | Moderate exploration |
| Mutation magnitude | 0.05 (std dev) | Small perturbations |
| Selection: top K | 10 | Top 20% survive |
| Display framerate | 30 FPS | Smooth canvas animation |
| Simulation tick rate | 60 ticks/sec (adjustable) | Fast enough to watch flow, slow enough to observe |

---

## 10. Implementation Plan

### Phase 1: Model and Minimal Simulation (Week 1-2)

**Goal:** World ticks, agents loop, particles flow between agents. No visualization, no GA, no AI. Console output only.

**Tasks:**
1. Initialize uv project, set up `pyproject.toml` with FastAPI, uvicorn, pytest, ruff dependencies.
2. Implement all model dataclasses: Agent, Link, Particle, Label, GenomeSchema, World.
3. Implement agent loop stepper: `step_agent()` advancing OODA phases.
4. Implement simulation tick: `tick_world()` stepping all agents, advancing particles, delivering arrivals, garbage collecting.
5. Implement sandwich shop corpus: hardcoded agents, links, labels, policies, external input generator.
6. Write tests verifying: agents cycle through OODA phases correctly, particles traverse links and arrive in input buffers, external inputs spawn at configured rates, agent policies produce expected outputs.
7. Console runner that ticks the world N times and prints state summaries.

**Deliverable:** `uv run python -m loopengine.corpora.sandwich_shop` runs the simulation headlessly and prints tick-by-tick state.

### Phase 2: Frame Projection and Visualization (Week 3-4)

**Goal:** The breathing canvas. Agents visible, particles flowing, links swaying.

**Tasks:**
1. Implement force-directed layout engine. Agents settle into positions based on link structure.
2. Implement Frame projector: `project(world) → Frame`.
3. Implement FastAPI server with WebSocket frame streaming.
4. Build frontend: HTML canvas, WebSocket connection, rendering loop.
5. Implement agent rendering (amoeba shapes, breathing, glow).
6. Implement link rendering (bezier curves, sway, thickness).
7. Implement particle rendering (colored dots, trails).
8. Implement label region rendering (convex hull clouds).
9. Implement interaction: hover tooltips, click selection, zoom, pan, play/pause/speed.

**Deliverable:** Open `localhost:8000` in a browser and watch the sandwich shop breathe.

### Phase 3: Genetic Algorithm (Week 5-6)

**Goal:** Evolve optimal genomes for each role.

**Tasks:**
1. Implement GA engine: population initialization, fitness evaluation, selection, crossover, mutation.
2. Implement fitness functions for Tom, Alex, Maria.
3. Implement GA run endpoint and WebSocket commands.
4. Add a simple GA progress visualization to the frontend (generation counter, best fitness graph).
5. Run experiments: evolve Tom with different Marias, evolve Alex with different customer volumes.
6. Write tests verifying: fitness improves over generations, genome values stay in valid ranges, crossover and mutation produce valid offspring.

**Deliverable:** `POST /api/ga/run` evolves a role and returns the best genome. Visible fitness improvement over generations.

### Phase 4: AI Genome Discovery (Week 7-8)

**Goal:** AI discovers genome schemas autonomously.

**Tasks:**
1. Implement discoverer module: takes system description, calls Claude API, parses GenomeSchema from response.
2. Design and test the discovery prompt template.
3. Implement schema migration: when a new schema arrives, existing genomes are gracefully updated.
4. Implement periodic rediscovery triggers (manual, schedule, GA stagnation detection).
5. Implement input randomization based on AI-inferred flexibility parameter.
6. Add discovery UI: button to trigger, display of current schemas, diff when schemas change.

**Deliverable:** Click "Discover Genomes" in the UI, see the AI propose a schema, watch it get applied to agents.

### Phase 5: Polish and Extensibility (Week 9-10)

**Goal:** Robustness, documentation, second corpus.

**Tasks:**
1. Add a second corpus (e.g., a small software team: PM, two developers, a designer) to validate universality.
2. Corpus loader UI: dropdown to switch between scenarios.
3. Performance optimization: frame rate, particle count limits, spatial indexing for large agent counts.
4. Error handling, logging, graceful degradation.
5. Documentation: README, architecture guide, how to create a new corpus.
6. Package for distribution via `uv`.

**Deliverable:** A polished, documented v0.1 that can load any corpus and simulate, visualize, and evolve it.

---

## 11. Success Criteria

### 11.1 Functional

- [ ] Three agents (Maria, Tom, Alex) loop independently at different speeds.
- [ ] Particles flow visibly between agents along links.
- [ ] External customer arrivals drive the system with a configurable lunch rush pattern.
- [ ] The GA evolves measurably better genomes over 100 generations for each role.
- [ ] AI genome discovery produces meaningful, role-appropriate trait schemas from a system description.
- [ ] The canvas visualization displays agents, links, particles, and label regions with breathing animation.
- [ ] Hover and click interactions reveal agent details.
- [ ] A second corpus can be loaded without changing any framework code.

### 11.2 Visual / Experiential

- [ ] At a glance, the viewer can tell which agent is busiest (particle density, glow, breathing rate).
- [ ] The lunch rush is visible as a surge in particle flow.
- [ ] Hierarchical relationships are legible from flow direction and vertical positioning without reading labels.
- [ ] The visualization feels organic and alive, not mechanical or static.

### 11.3 Technical

- [ ] All backend code passes `ruff` linting with no warnings.
- [ ] Test coverage above 80% for model and engine layers.
- [ ] Simulation runs at 60+ ticks/sec with 3 agents and up to 50 active particles.
- [ ] Frame streaming maintains 30 FPS to the frontend.
- [ ] The project installs and runs cleanly with `uv sync && uv run loopengine`.

---

## 12. Open Questions

These are decisions deferred to implementation time:

1. **Policy representation:** For v0.1, policies are hand-written Python functions. Should v0.2 use evolved decision trees, neural networks, or LLM-generated policies?

2. **Multi-objective fitness:** Maria's fitness involves throughput, margin, and waste. Should these be combined into a single scalar (weighted sum) or should the GA use Pareto frontier optimization (NSGA-II)?

3. **Schema versioning:** When the AI proposes a new genome schema, how much continuity should be maintained? Should running GA populations be migrated mid-evolution, or should evolution restart with the new schema?

4. **Agent memory:** The current model has `internal_state` as a dict, which resets each simulation run. Should agents accumulate persistent memory across OODA revolutions? Across simulation runs?

5. **Link evolution:** Can the GA evolve link properties (e.g., how much autonomy a supervisor grants) in addition to agent genomes? This would let the framework discover optimal organizational structures, not just optimal individuals.

6. **Scale limits:** At what agent count does the force-directed layout or frame projection become a bottleneck? When should we switch to spatial hashing, quadtrees, or GPU-accelerated rendering?

7. **Determinism:** Should simulation runs be deterministic given a seed? This aids reproducibility and debugging but constrains the external input generator.

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **Agent** | An individual entity that senses, decides, and acts within a system. Always a single individual, never a group. |
| **Genome** | A dictionary of trait names to float values encoding an agent's capabilities and tendencies. The substrate that the GA evolves. |
| **GenomeSchema** | The AI-discovered definition of what traits are meaningful for a given role. Includes trait names, descriptions, ranges, and categories. |
| **Label** | A shared identifier carried by agents who share context, constraints, or proximity. Not an actor. |
| **Link** | A typed, directional connection between two agents. Carries properties defining the relationship (authority, autonomy, flow types). |
| **Particle** | A discrete unit of flow — an order, a product, a directive, a report — traveling along a link from one agent to another. |
| **Policy** | The decision-making function that maps an agent's sensed inputs, genome, and internal state to actions (output particles). |
| **OODA Loop** | Sense → Orient → Decide → Act. The universal agent cycle. Every agent runs this loop at its own period. |
| **Loop Period** | How many simulation ticks one full OODA revolution takes. Short periods = fast agents. Long periods = slow, strategic agents. |
| **Frame** | A visual snapshot of the world state, containing positions, sizes, colors, and animation phases for all visible elements. Consumed by the frontend renderer. |
| **World** | The container holding all agents, links, particles, labels, and schemas. The source of truth for the simulation. |
| **Fitness** | A scalar measure of how well an agent performs its role. Defined per-role. Used by the GA for selection. |
| **Flexibility** | How much input variance an agent's role demands. Discovered by the AI. Controls input randomization during evolution and simulation. |

---

## Appendix B: Example AI Discovery Prompt

The following is a template for the prompt sent to the Claude API during genome discovery:

```
You are analyzing an organizational system to discover the meaningful
dimensions of variation for agents in each role.

SYSTEM DESCRIPTION:
{system_description_json}

For each role in this system, identify the traits that would meaningfully
affect an agent's performance. Consider:
- What cognitive, physical, social, and temperamental traits matter?
- What skills or aptitudes differentiate good from poor performance?
- What tendencies or biases affect decision-making in this role?
- What traits affect how this agent interacts with linked agents?

For each trait, provide:
- name: a snake_case identifier
- description: what this trait represents
- category: one of "physical", "cognitive", "social", "temperamental", "skill"
- min_val: minimum value (typically 0.0)
- max_val: maximum value (typically 1.0)

Also provide a flexibility_score (0.0 to 1.0) for each role indicating
how much input variance the role typically faces.

Respond with valid JSON only, no additional commentary.
```

---

*End of document.*
