# Creating Corpora

This guide walks through creating new corpora (simulation scenarios) for LoopEngine.

## Overview

A corpus is a complete simulation scenario containing:

- **Agents**: Individual entities with roles, genomes, and policies
- **Links**: Typed connections between agents
- **Labels**: Shared contexts for groups of agents
- **External Inputs**: Event sources from outside the system

## Quick Start

Create a minimal corpus in 5 steps:

```bash
# 1. Create corpus directory
mkdir -p src/loopengine/corpora/my_corpus

# 2. Create __init__.py with create_world() function
# (see template below)

# 3. Register in server/app.py AVAILABLE_CORPORA

# 4. Restart server
uv run flask --app app.main run --port 8000

# 5. Load via API or dropdown
curl -X POST http://localhost:8000/api/world/load_corpus \
  -H "Content-Type: application/json" \
  -d '{"corpus_name": "my_corpus"}'
```

## Required Components

### 1. Agents

Agents are individuals with OODA loops. Each needs:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier (lowercase, no spaces) |
| `name` | str | Display name |
| `role` | str | Role type (for genome schemas) |
| `genome` | dict[str, float] | Trait values 0.0-1.0 |
| `labels` | set[str] | Label names this agent carries |
| `loop_period` | int | Ticks per OODA cycle (higher = slower) |
| `policy` | Callable | Decision function (see Policy Template) |

**Example:**

```python
from loopengine.model import Agent

agents["barista"] = Agent(
    id="barista",
    name="Sam",
    role="barista",
    genome={
        "speed": 0.7,
        "accuracy": 0.8,
        "friendliness": 0.6,
    },
    labels={"CoffeeShop", "FrontLine"},
    loop_period=30,  # Fast cycle for customer-facing
    policy=barista_policy,
)
```

**Genome Design Tips:**

- Traits should be independent dimensions of variation
- Values 0.0-1.0 representing low-high capability
- Choose traits that affect observable behavior
- 3-5 traits per role is typical

### 2. Links

Links connect agents with typed, directional relationships.

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `source_id` | str | Origin agent id |
| `dest_id` | str | Destination agent id |
| `link_type` | LinkType | HIERARCHICAL, PEER, SERVICE, COMPETITIVE |
| `properties` | dict | Flow types, bandwidth, authority scope |

**Link Types:**

| Type | Use Case |
|------|----------|
| `HIERARCHICAL` | Authority relationships (manager → worker) |
| `PEER` | Coordination between equals |
| `SERVICE` | Provider/consumer relationships |
| `COMPETITIVE` | Competitive relationships |

**Example:**

```python
from loopengine.model import Link, LinkType

links["manager_to_barista"] = Link(
    id="manager_to_barista",
    source_id="manager",
    dest_id="barista",
    link_type=LinkType.HIERARCHICAL,
    properties={
        "authority_scope": ["scheduling", "quality_standards"],
        "autonomy_granted": 0.6,
        "flow_types": ["directive", "feedback"],
    },
)

links["barista_to_manager"] = Link(
    id="barista_to_manager",
    source_id="barista",
    dest_id="manager",
    link_type=LinkType.HIERARCHICAL,
    properties={
        "flow_types": ["status_report", "supply_alert"],
    },
)
```

**Bidirectional Relationships:**

Model as two separate links (one each direction).

### 3. Labels

Labels define shared contexts without acting themselves.

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Label identifier |
| `context.constraints` | list[str] | Rules agents must follow |
| `context.resources` | list[str] | Available resources |
| `context.norms` | list[str] | Expected behaviors |

**Example:**

```python
from loopengine.model import Label, LabelContext

labels["CoffeeShop"] = Label(
    name="CoffeeShop",
    context=LabelContext(
        constraints=["health_code", "max_capacity_50"],
        resources=["espresso_machine", "grinder", "POS_system"],
        norms=["greet_customers", "clean_as_you_go"],
    ),
)

labels["FrontLine"] = Label(
    name="FrontLine",
    context=LabelContext(
        constraints=["customer_facing_appearance"],
        norms=["friendly_demeanor", "no_phone_use"],
    ),
)
```

**Label Assignment:**

Agents reference labels by name in their `labels` set.

### 4. External Inputs

External inputs spawn particles from outside the system.

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Input identifier |
| `target_agent_id` | str | Who receives particles |
| `rate` | float | Average particles per tick |
| `variance` | float | Randomness in rate (0.0-1.0) |
| `particle_type` | str | Type of spawned particles |
| `payload_generator` | Callable | Function returning payload dict |
| `schedule` | Callable | Function(tick) → rate multiplier |

**Example:**

```python
from loopengine.model import ExternalInput

def generate_coffee_order() -> dict[str, Any]:
    return {
        "drink": random.choice(["latte", "espresso", "cappuccino"]),
        "size": random.choice(["small", "medium", "large"]),
        "extras": random.sample(["oat_milk", "extra_shot"], k=random.randint(0, 1)),
    }

def morning_rush(tick: int) -> float:
    """Double rate during morning rush (ticks 100-300)."""
    if 100 <= tick <= 300:
        return 2.0
    return 1.0

external_inputs = [
    ExternalInput(
        name="customer_orders",
        target_agent_id="barista",
        rate=0.05,  # ~3 per minute at 60 ticks/sec
        variance=0.3,
        particle_type="coffee_order",
        payload_generator=generate_coffee_order,
        schedule=morning_rush,
    ),
]
```

### 5. Policies

Policies are the decision logic for agents. They receive inputs and produce outputs.

**Signature:**

```python
def my_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Process inputs and return output particles."""
    ...
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `sensed_inputs` | Particles received this OODA cycle |
| `genome` | Agent's trait values (read-only) |
| `internal_state` | Persistent state dict (read-write) |

**Return:**

List of `Particle` objects to emit.

## Policy Template

```python
from loopengine.model import Particle

def barista_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Barista policy: make drinks and serve customers.

    SENSE: Read customer orders
    ORIENT: Check current workload
    DECIDE: Plan drink preparation
    ACT: Emit finished drinks
    """
    outputs: list[Particle] = []

    # Initialize tracking metrics
    if "drinks_made" not in internal_state:
        internal_state["drinks_made"] = 0
    if "queue" not in internal_state:
        internal_state["queue"] = []

    # Process each input particle
    for particle in sensed_inputs:
        if particle.particle_type == "coffee_order":
            # Queue the order
            internal_state["queue"].append(particle)

        elif particle.particle_type == "directive":
            # Store directive for behavior modification
            internal_state["current_directive"] = particle.payload

    # Make drinks from queue (genome affects speed/quality)
    speed = genome.get("speed", 0.5)
    accuracy = genome.get("accuracy", 0.5)

    # Process based on speed trait (higher = more per cycle)
    drinks_to_make = max(1, int(speed * 2))

    for _ in range(min(drinks_to_make, len(internal_state["queue"]))):
        order = internal_state["queue"].pop(0)

        # Quality affected by accuracy trait
        quality = 0.5 + (accuracy * 0.5)

        outputs.append(
            Particle(
                id=f"drink_{order.id}",
                particle_type="finished_drink",
                payload={
                    "order": order.payload,
                    "quality": quality,
                    "barista": "barista",
                },
                source_id="barista",
                dest_id="customer",  # or external
                link_id="",  # empty for external destinations
            )
        )
        internal_state["drinks_made"] += 1

    return outputs
```

## Complete Corpus Template

```python
"""My Corpus: Description of the scenario."""

from __future__ import annotations

import random
from typing import Any

from loopengine.model import (
    Agent,
    ExternalInput,
    Label,
    LabelContext,
    Link,
    LinkType,
    Particle,
    World,
)


# ============================================================
# POLICIES
# ============================================================

def agent_a_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Agent A's decision logic."""
    outputs: list[Particle] = []

    for particle in sensed_inputs:
        if particle.particle_type == "input_type":
            # Process and emit output
            outputs.append(
                Particle(
                    id=f"output_{particle.id}",
                    particle_type="output_type",
                    payload={"processed": True},
                    source_id="agent_a",
                    dest_id="agent_b",
                    link_id="a_to_b",
                )
            )

    return outputs


def agent_b_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Agent B's decision logic."""
    outputs: list[Particle] = []
    # ... implementation ...
    return outputs


# ============================================================
# CREATION FUNCTIONS
# ============================================================

def create_agents() -> dict[str, Agent]:
    """Create all agents for the scenario."""
    agents = {}

    agents["agent_a"] = Agent(
        id="agent_a",
        name="Agent A",
        role="role_a",
        genome={"trait_1": 0.7, "trait_2": 0.5},
        labels={"GlobalLabel", "SpecificLabel"},
        loop_period=60,
        policy=agent_a_policy,
    )

    agents["agent_b"] = Agent(
        id="agent_b",
        name="Agent B",
        role="role_b",
        genome={"trait_1": 0.6, "trait_2": 0.8},
        labels={"GlobalLabel"},
        loop_period=40,
        policy=agent_b_policy,
    )

    return agents


def create_links() -> dict[str, Link]:
    """Create all links between agents."""
    links = {}

    links["a_to_b"] = Link(
        id="a_to_b",
        source_id="agent_a",
        dest_id="agent_b",
        link_type=LinkType.SERVICE,
        properties={"flow_types": ["output_type"], "bandwidth": 1.0},
    )

    return links


def create_labels() -> dict[str, Label]:
    """Create all labels for shared contexts."""
    labels = {}

    labels["GlobalLabel"] = Label(
        name="GlobalLabel",
        context=LabelContext(
            constraints=["constraint_1"],
            resources=["resource_1"],
            norms=["norm_1"],
        ),
    )

    return labels


def generate_input_payload() -> dict[str, Any]:
    """Generate random input payload."""
    return {"value": random.random()}


def create_external_inputs() -> list[ExternalInput]:
    """Create external event sources."""
    return [
        ExternalInput(
            name="external_events",
            target_agent_id="agent_a",
            rate=0.05,
            variance=0.2,
            particle_type="input_type",
            payload_generator=generate_input_payload,
            schedule=lambda tick: 1.0,
        ),
    ]


def create_world() -> World:
    """Create the complete world for this corpus."""
    world = World()
    world.agents = create_agents()
    world.links = create_links()
    world.labels = create_labels()
    world.external_inputs = create_external_inputs()
    return world
```

## Registering Your Corpus

Add to `src/loopengine/server/app.py`:

```python
# At the top, add import
from loopengine.corpora.my_corpus import create_world as create_my_corpus_world

# In SimulationState class, add to AVAILABLE_CORPORA
AVAILABLE_CORPORA: ClassVar[dict[str, tuple[str, Any]]] = {
    "sandwich_shop": ("Sandwich Shop", create_sandwich_shop_world),
    "software_team": ("Software Team", create_software_team_world),
    "my_corpus": ("My Corpus", create_my_corpus_world),  # Add this line
}
```

## Testing Checklist

Before considering your corpus complete:

- [ ] **Syntax**: No import errors (`python -c "from loopengine.corpora.my_corpus import create_world"`)
- [ ] **World Creation**: `create_world()` returns valid World
- [ ] **Agent IDs**: All agent IDs are unique
- [ ] **Link Validation**: All link source/dest IDs reference existing agents
- [ ] **Label References**: All agent labels exist in labels dict
- [ ] **Policy Functions**: All policies handle expected particle types
- [ ] **External Inputs**: Target agents exist
- [ ] **Type Hints**: All functions have proper type hints
- [ ] **Docstrings**: Module, functions, and policies documented
- [ ] **Registration**: Corpus added to AVAILABLE_CORPORA

**Verification Script:**

```python
from loopengine.corpora.my_corpus import create_world

# Test world creation
world = create_world()

# Verify agents
assert len(world.agents) > 0, "No agents defined"
for agent_id, agent in world.agents.items():
    assert agent.id == agent_id, f"Agent ID mismatch: {agent_id}"
    assert agent.policy is not None, f"Agent {agent_id} has no policy"

# Verify links reference valid agents
for link_id, link in world.links.items():
    assert link.source_id in world.agents, f"Link {link_id} source not found"
    assert link.dest_id in world.agents, f"Link {link_id} dest not found"

# Verify labels referenced by agents exist
for agent in world.agents.values():
    for label_name in agent.labels:
        assert label_name in world.labels, f"Label {label_name} not found"

# Verify external inputs target valid agents
for ext_input in world.external_inputs:
    assert ext_input.target_agent_id in world.agents, \
        f"External input targets unknown agent: {ext_input.target_agent_id}"

print("All validations passed!")
```

## Common Pitfalls

### 1. Mismatched IDs

```python
# WRONG: ID in dict doesn't match agent.id
agents["alice"] = Agent(id="alice_agent", ...)

# CORRECT: Match the ID
agents["alice"] = Agent(id="alice", ...)
```

### 2. Missing Bidirectional Links

```python
# WRONG: Only one direction
links["manager_to_worker"] = Link(...)

# CORRECT: Both directions for full communication
links["manager_to_worker"] = Link(source_id="manager", dest_id="worker", ...)
links["worker_to_manager"] = Link(source_id="worker", dest_id="manager", ...)
```

### 3. Referencing Non-Existent Labels

```python
# WRONG: Agent references label that doesn't exist
agents["bob"] = Agent(labels={"Team", "Engineering"}, ...)
labels = {"Team": Label(...)}  # Missing "Engineering"

# CORRECT: Define all referenced labels
labels = {
    "Team": Label(...),
    "Engineering": Label(...),
}
```

### 4. Empty Output Lists in Policies

```python
# WRONG: No return statement or returning None
def policy(sensed_inputs, genome, internal_state):
    for particle in sensed_inputs:
        process(particle)
    # Forgot to return!

# CORRECT: Always return list (even if empty)
def policy(sensed_inputs, genome, internal_state):
    outputs = []
    for particle in sensed_inputs:
        outputs.extend(process(particle))
    return outputs
```

### 5. Incorrect Link IDs in Particles

```python
# WRONG: Link ID doesn't exist
Particle(link_id="nonexistent_link", ...)

# CORRECT: Use actual link ID or empty string for external
Particle(link_id="manager_to_worker", ...)  # Valid link
Particle(link_id="", ...)  # External destination
```

### 6. Forgetting Internal State Initialization

```python
# WRONG: Assumes key exists
def policy(sensed_inputs, genome, internal_state):
    internal_state["count"] += 1  # KeyError!

# CORRECT: Initialize first
def policy(sensed_inputs, genome, internal_state):
    if "count" not in internal_state:
        internal_state["count"] = 0
    internal_state["count"] += 1
```

### 7. Loop Period Too Short/Long

```python
# WRONG: All agents same period (unrealistic)
Agent(loop_period=60, ...)  # Manager
Agent(loop_period=60, ...)  # Worker
Agent(loop_period=60, ...)  # Another worker

# CORRECT: Vary by role complexity
Agent(loop_period=200, ...)  # Manager (strategic, slower cycle)
Agent(loop_period=40, ...)   # Worker (tactical, faster cycle)
Agent(loop_period=60, ...)   # Designer (medium complexity)
```

### 8. Rate Multiplier Returns Zero

```python
# WRONG: Schedule returns 0, no events spawn
def schedule(tick: int) -> float:
    if tick < 100:
        return 0  # Nothing happens!
    return 1.0

# CORRECT: Use low values instead of zero if needed
def schedule(tick: int) -> float:
    if tick < 100:
        return 0.1  # Slow start
    return 1.0
```

## Next Steps

After creating your corpus:

1. **Test manually**: Load via UI and observe simulation
2. **Add fitness functions**: If you want GA evolution (see `engine/fitness.py`)
3. **Create genome schemas**: For AI-discovered traits (see `discovery/`)
4. **Profile performance**: Run with `experiments/profile_simulation.py`

## Reference Examples

Study the existing corpora for patterns:

- `src/loopengine/corpora/sandwich_shop/`: Simple 3-agent hierarchy
- `src/loopengine/corpora/software_team/`: 4-agent with peer relationships

See also:
- `docs/architecture.md`: Full system architecture
- `README.md`: Quick start and API reference
