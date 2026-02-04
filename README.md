# LoopEngine

A simulation framework and visual explorer for modeling any agent in any system using a universal schema.

LoopEngine treats every agent—from line cooks to CEOs—as having the same formal structure: inputs, outputs, a genome of capabilities, behavioral policies, typed links to other agents, and a looping sense-orient-decide-act cycle. The framework evolves agent genomes using genetic algorithms, uses AI to discover genome properties, and renders the living system as an organic, breathing canvas visualization.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/loopengine.git
cd loopengine

# Install dependencies
uv sync
```

### Running the Simulation

```bash
# Terminal 1: Start the backend server
uv run uvicorn loopengine.server.app:app --port 8000

# Terminal 2: Serve the frontend (from project root)
python -m http.server 8080

# Open the visualizer
open http://localhost:8080/frontend/index.html
```

The browser will display an interactive canvas showing agents, their connections, and particles flowing between them. Use the controls to play/pause the simulation, adjust speed, and switch between corpora.

**Note**: The frontend connects to `ws://localhost:8000` for WebSocket streams, so ensure the backend is running on port 8000.

## Features

- **Universal Agent Model**: Every agent has the same structure—genome, policies, links, labels, and OODA loop phases
- **Visual Simulation**: Real-time canvas rendering with pulsing agents, flowing particles, and breathing links
- **Multiple Corpora**: Switch between scenarios (Sandwich Shop, Software Team) via dropdown
- **Genetic Algorithm**: Evolve agent genomes to optimize fitness functions
- **AI Discovery**: Use LLMs to discover meaningful genome traits for new roles
- **WebSocket Streaming**: 30 FPS frame updates for smooth visualization

## Visualization

The canvas visualization shows:

```
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │         [Maria]◄───────────────────►[Tom]                │
    │           Owner                    Sandwich Maker        │
    │             │         ○ ○ ○            │                 │
    │             │       (particles)        │                 │
    │             │         ○ ○              │                 │
    │             ▼                          ▼                 │
    │                      [Alex]                              │
    │                      Cashier                             │
    │                                                          │
    │   ┌─────────────────────────────────────────────────┐    │
    │   │ ▶ Play  ⏸ Pause  Speed: [1.0x ▼]  Corpus: [▼]  │    │
    │   └─────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────┘
```

- **Agents** appear as pulsing circles with labels showing name and role
- **Links** connect agents as animated curves that "breathe" based on flow
- **Particles** travel along links representing information/resources in transit
- **Labels** form translucent cloud regions grouping related agents
- **Controls** at bottom for play/pause, speed adjustment, and corpus selection

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                    │
│   Canvas Renderer ← Frame Snapshots ← WebSocket          │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────┐
│                    BACKEND (Python)                      │
│   Model Layer → Loop Engine → Frame Projector            │
│   AI Discovery → GA Engine                               │
└─────────────────────────────────────────────────────────┘
```

### Layers

| Layer | Description |
|-------|-------------|
| **Model** | Domain entities: Agent, Link, Particle, Label, World |
| **Engine** | Simulation tick loop, OODA phases, genetic algorithm |
| **Projection** | Transforms World state into renderable Frames |
| **Server** | FastAPI with REST API and WebSocket streaming |
| **Frontend** | Vanilla JS canvas visualization |

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/world` | Get current world state (tick, speed, agent/link counts) |
| GET | `/api/corpora` | List available corpora |
| GET | `/api/agents` | Get all agents |
| GET | `/api/agents/{id}` | Get specific agent |
| GET | `/api/links` | Get all links |
| GET | `/api/schemas` | Get genome schemas |
| POST | `/api/world/reset` | Reset world to initial state |
| POST | `/api/world/load_corpus?corpus_name=` | Load a named corpus |
| POST | `/api/world/pause` | Pause simulation |
| POST | `/api/world/play` | Resume simulation |
| POST | `/api/world/speed?speed=` | Set speed multiplier (0.1-10.0) |
| POST | `/api/ga/run` | Start genetic algorithm evolution |
| GET | `/api/ga/status/{job_id}` | Get GA job status |
| POST | `/api/discovery/run` | Start AI genome discovery |
| GET | `/api/discovery/status/{job_id}` | Get discovery job status |
| GET | `/health` | Health check |

### WebSocket Endpoints

#### `/ws/frames` - Frame Streaming

Streams Frame objects at ~30 FPS containing all visual elements:

```json
{
  "tick": 1234,
  "time": 45.6,
  "agents": [...],
  "links": [...],
  "particles": [...],
  "label_regions": [...]
}
```

#### `/ws/control` - Control Commands

Send commands to control the simulation:

```json
// Play/Pause
{"type": "play"}
{"type": "pause"}

// Set speed
{"type": "set_speed", "speed": 2.0}

// Reset
{"type": "reset"}

// Genetic Algorithm
{"type": "start_ga", "role": "sandwich_maker", "generations": 100}
{"type": "stop_ga"}
{"type": "get_ga_status"}
```

Server broadcasts progress updates:

```json
{"type": "ga_progress", "job_id": "...", "generation": 50, "best_fitness": 0.85}
{"type": "ga_complete", "job_id": "...", "best_genome": {...}}
```

## Available Corpora

### Sandwich Shop

A small sandwich shop with three employees demonstrating hierarchical and service relationships:

- **Maria** (Owner): Sets standards, monitors operations
- **Tom** (Sandwich Maker): Prepares orders, manages inventory
- **Alex** (Cashier): Takes orders, handles payments

### Software Team

A software development team with typical tech company dynamics:

- **PM** (Product Manager): Assigns tasks, balances priorities
- **Dev1/Dev2** (Developers): Write code, review each other's work
- **Designer**: Creates specs and design assets

## Development

### Running Tests

```bash
uv run pytest
```

### Linting and Formatting

```bash
uv run ruff check .
uv run ruff format .
```

### Project Structure

```
loopengine/
├── src/loopengine/
│   ├── model/          # Domain entities (Agent, Link, World, etc.)
│   ├── engine/         # Simulation loop, GA, fitness functions
│   ├── projection/     # World → Frame transformation
│   ├── server/         # FastAPI app, REST & WebSocket endpoints
│   ├── discovery/      # AI-powered genome discovery
│   ├── behaviors/      # AI behavior generation
│   ├── corpora/        # Scenario definitions
│   └── api/            # Additional API routers
├── frontend/           # Browser visualization
├── tests/              # Test suite
└── prd/                # Product requirements
```

## Core Concepts

### Agents

Individual actors in the system. Each agent has:
- **Genome**: Numeric traits that influence behavior
- **Role**: Defines what genome schema applies
- **Labels**: Shared context tags (e.g., "Kitchen Staff")
- **OODA Loop**: Cycles through Observe → Orient → Decide → Act phases

### Links

Typed, directional connections between agents:
- **HIERARCHICAL**: Authority relationships (manager → report)
- **PEER**: Equal collaboration (coworker ↔ coworker)
- **SERVICE**: Request/response patterns (customer → provider)

### Particles

Information or resources flowing through links. Each particle carries a payload and traverses from source to destination agent.

### Labels

Shared context that groups agents. Labels define constraints, norms, and resources that apply to all members.

## License

MIT
