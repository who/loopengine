# PRD: Dynamic AI-Driven Agent Behaviors

## Metadata
- **Feature ID**: loopengine-b4y
- **Project Type**: Full-Stack
- **Created**: 2026-02-03
- **Author**: Claude (one-shot generation)
- **Generation Mode**: One-shot (no interview)
- **Interview Confidence**: N/A (skipped)

## Overview

### Problem Statement
Currently, loopengine requires users to write custom Python code for each simulation domain (e.g., sandwich shop behaviors are hardcoded in Python files). This creates a significant barrier for users who want to prototype simulations in new domains like flower shops, speaker warehouses, or other business scenarios. Each new domain requires programming expertise and substantial development time to define agent behaviors, decision logic, and interactions. Users without Python skills cannot experiment with the simulation framework, and even experienced developers spend unnecessary time writing boilerplate behavior code for each new scenario.

### Proposed Solution
Implement a dynamic agent behavior system that uses LLM APIs at runtime to determine agent actions based on context. When a user defines a domain (describing what kind of business or system they want to simulate), the system queries a configured LLM to generate appropriate agent behaviors, decision logic, and interactions on-the-fly. This enables rapid prototyping of any domain simulation without writing custom Python code. Users configure their API token (e.g., Claude API key) in a `.env` file, describe their domain in natural language, and the system handles behavior generation automatically.

### Success Metrics
- Users can create a functional simulation for a new domain in under 5 minutes without writing Python code
- At least 80% of generated agent behaviors produce logically consistent simulation outcomes
- System supports at least 3 different LLM providers (Claude, OpenAI, local models)
- Average LLM response latency under 2 seconds per agent decision

## Background & Context
Loopengine is a simulation framework and visual explorer for modeling agents in any system using a universal schema. The current implementation demonstrates capabilities with a sandwich shop example that uses hardcoded Python behaviors. While this proves the concept, it limits the framework's utility to users who can write Python code. By leveraging LLMs to generate behavior dynamically, loopengine can become a truly universal simulation tool where users describe what they want to simulate in plain language and immediately see results.

## Users & Personas

### Primary Persona: Domain Expert
- **Role**: Business analyst, operations manager, or subject matter expert
- **Goals**: Quickly prototype and visualize how their business processes work; test "what-if" scenarios
- **Pain Points**: Cannot write Python code; wants to simulate their domain without developer involvement
- **Technical Level**: Beginner (no programming experience)

### Secondary Persona: Simulation Developer
- **Role**: Developer using loopengine for custom simulations
- **Goals**: Accelerate prototyping of new domains; reduce boilerplate code for behavior logic
- **Technical Level**: Expert (can fall back to custom Python when needed)

### User Journeys

#### Journey 1: Creating a New Domain Simulation
1. User navigates to loopengine UI or configuration file
2. User describes their domain in natural language (e.g., "A flower shop with customers ordering bouquets, florists preparing arrangements, and delivery drivers")
3. System generates initial agent types, behaviors, and interaction rules
4. User launches simulation and observes agents acting according to generated behaviors
5. User refines domain description or adds constraints as needed

#### Journey 2: Refining Agent Behavior
1. User observes simulation running with AI-generated behaviors
2. User notices an agent acting illogically for the domain
3. User provides feedback or additional constraints in natural language
4. System regenerates or adjusts the specific behavior
5. Simulation continues with improved behavior logic

## Requirements

### Functional Requirements
[P0] FR-001: The system shall accept a natural language domain description and generate appropriate agent types
[P0] FR-002: The system shall query a configured LLM at runtime to determine agent actions based on current simulation state
[P0] FR-003: Users shall be able to configure LLM API credentials via environment variables (.env file)
[P0] FR-004: The system shall provide a fallback mechanism when LLM is unavailable (cached behaviors or error state)
[P1] FR-005: Users shall be able to provide additional behavioral constraints in natural language
[P1] FR-006: The system shall support multiple LLM providers (Claude, OpenAI, local Ollama models)
[P1] FR-007: Users shall be able to view and export generated behaviors for inspection
[P2] FR-008: The system shall cache frequently-used behavior patterns to reduce API calls
[P2] FR-009: Users shall be able to "pin" good behaviors to prevent regeneration

### Non-Functional Requirements
[P0] NFR-001: LLM queries shall complete within 3 seconds for 95% of requests
[P0] NFR-002: API credentials shall be stored securely and never exposed in logs or UI
[P1] NFR-003: The system shall gracefully degrade when LLM rate limits are hit
[P1] NFR-004: The system shall support at least 50 agents making concurrent decisions
[P2] NFR-005: Generated behaviors shall be deterministic given the same random seed and context

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client Layer                        │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Frontend Application                    ││
│  │         (Existing loopengine UI)                    ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                           │
                           │ HTTPS / WebSocket
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     Server Layer                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Flask Backend                           ││
│  │    ┌─────────────────────────────────────────┐      ││
│  │    │       AI Behavior Engine (NEW)          │      ││
│  │    │  - Domain Parser                        │      ││
│  │    │  - Prompt Builder                       │      ││
│  │    │  - LLM Client Adapter                   │      ││
│  │    │  - Behavior Cache                       │      ││
│  │    └─────────────────────────────────────────┘      ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│    External LLMs      │   │    Behavior Cache    │
│  - Claude API         │   │  - Recent decisions  │
│  - OpenAI API         │   │  - Domain schemas    │
│  - Ollama (local)     │   │  - Pinned behaviors  │
└──────────────────────┘   └──────────────────────┘
```

### Backend Architecture

**Framework**: Flask (existing)

**New Structure**:
```
src/
├── engine/                    # Existing simulation engine
├── behaviors/                 # NEW: AI behavior system
│   ├── __init__.py
│   ├── ai_behavior_engine.py  # Main orchestrator
│   ├── domain_parser.py       # Parse natural language domains
│   ├── prompt_builder.py      # Build LLM prompts from context
│   ├── llm_client.py          # Adapter for multiple LLM providers
│   ├── behavior_cache.py      # Cache and pin behaviors
│   └── providers/
│       ├── __init__.py
│       ├── claude.py          # Claude API client
│       ├── openai.py          # OpenAI API client
│       └── ollama.py          # Local Ollama client
└── ...
```

**Key Technologies**:
| Category | Choice | Rationale |
|----------|--------|-----------|
| LLM Client | anthropic/openai SDKs | Official SDKs for reliability |
| Caching | In-memory dict + optional Redis | Simple for MVP, scalable later |
| Config | python-dotenv | Standard .env file handling |

### API Contract

**Base URL**: `/api/v1`

**New Endpoints**:

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | /domains | Create/update domain description | No |
| GET | /domains/:id | Get domain configuration | No |
| POST | /behaviors/generate | Generate behavior for agent context | No |
| GET | /behaviors/cache | List cached behaviors | No |
| POST | /behaviors/pin | Pin a specific behavior | No |
| DELETE | /behaviors/cache | Clear behavior cache | No |

**Request/Response Format**:
```json
// POST /behaviors/generate
// Request
{
  "domain_id": "flower-shop",
  "agent_type": "florist",
  "context": {
    "current_state": "idle",
    "pending_orders": 3,
    "available_flowers": ["roses", "tulips"],
    "time_of_day": "morning"
  }
}

// Response
{
  "data": {
    "action": "prepare_order",
    "parameters": {
      "order_id": "order_001",
      "priority": "high"
    },
    "reasoning": "Multiple orders pending, starting with oldest"
  },
  "meta": {
    "cached": false,
    "latency_ms": 850,
    "provider": "claude"
  }
}
```

### LLM Integration Design

**Provider Configuration** (.env):
```
# Primary LLM provider
LLM_PROVIDER=claude  # claude | openai | ollama

# API Keys (only needed for cloud providers)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Ollama config (for local models)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# Behavior settings
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7
BEHAVIOR_CACHE_TTL=300  # seconds
```

**Prompt Structure**:
```
System: You are a behavior engine for a {domain_type} simulation.
Given the current state of an agent, determine their next action.

Domain Description: {domain_description}
Agent Type: {agent_type}
Agent Role: {agent_role}

Current Context:
{json_context}

Respond with a JSON object containing:
- action: The action the agent should take
- parameters: Any parameters for the action
- reasoning: Brief explanation (for debugging)
```

## Data Flow

### Behavior Generation Flow
```
1. Simulation tick triggers agent decision needed →
2. Engine calls AI Behavior Engine with context →
3. Check behavior cache for matching context →
4. If cache miss, build prompt from context →
5. Query configured LLM provider →
6. Parse and validate LLM response →
7. Cache the behavior →
8. Return action to simulation engine →
9. Agent executes action
```

### Domain Setup Flow
```
1. User enters domain description →
2. System parses description to extract:
   - Agent types
   - Resource types
   - Interaction patterns →
3. Generate initial domain schema →
4. Store domain configuration →
5. Simulation ready to run
```

## Deployment

### Environment Variables

**Backend**:
| Variable | Description | Example |
|----------|-------------|---------|
| LLM_PROVIDER | Which LLM to use | claude |
| ANTHROPIC_API_KEY | Claude API key | sk-ant-... |
| OPENAI_API_KEY | OpenAI API key | sk-... |
| OLLAMA_HOST | Local Ollama URL | http://localhost:11434 |
| LLM_TEMPERATURE | Response randomness | 0.7 |
| BEHAVIOR_CACHE_TTL | Cache duration (seconds) | 300 |

## Milestones & Phases

### Phase 1: Core LLM Integration
**Goal**: Basic behavior generation working with single provider
**Deliverables**:
- LLM client adapter for Claude API
- Basic prompt builder
- Integration with existing simulation engine
- .env configuration support

### Phase 2: Domain Parser & Multi-Provider
**Goal**: Natural language domain definition and provider flexibility
**Deliverables**:
- Domain description parser
- OpenAI provider support
- Ollama (local) provider support
- Behavior caching

### Phase 3: Polish & Robustness
**Goal**: Production-ready with good UX
**Deliverables**:
- Behavior pinning
- Cache management UI
- Rate limit handling
- Error recovery and fallbacks

## Epic Breakdown

### Epic: LLM Client Foundation
- **Requirements Covered**: FR-002, FR-003, FR-006
- **Tasks**:
  - [ ] Create LLM client adapter interface
  - [ ] Implement Claude API client
  - [ ] Implement OpenAI API client
  - [ ] Implement Ollama client
  - [ ] Add .env configuration loading

### Epic: Behavior Engine Core
- **Requirements Covered**: FR-001, FR-002, FR-004
- **Tasks**:
  - [ ] Create AI behavior engine orchestrator
  - [ ] Implement prompt builder with context formatting
  - [ ] Add response parsing and validation
  - [ ] Integrate with simulation engine tick cycle
  - [ ] Implement fallback mechanism

### Epic: Domain Parser
- **Requirements Covered**: FR-001, FR-005
- **Tasks**:
  - [ ] Create domain description parser
  - [ ] Extract agent types from natural language
  - [ ] Extract resource and interaction patterns
  - [ ] Store domain configuration

### Epic: Caching & Optimization
- **Requirements Covered**: FR-008, FR-009, NFR-001
- **Tasks**:
  - [ ] Implement behavior cache
  - [ ] Add cache key generation from context
  - [ ] Implement behavior pinning
  - [ ] Add cache invalidation logic

## Open Questions
- Should we support streaming responses for long-running behavior generation?
- How should we handle contradictory behaviors generated by the LLM?
- Should users be able to edit generated behaviors directly?
- What's the budget/cost threshold for LLM API usage per simulation?

## Out of Scope
- Training custom models for specific domains
- Fine-tuning LLMs on simulation data
- Real-time collaborative editing of domain descriptions
- Billing/payment integration for API usage
- Mobile-specific UI for domain configuration
- Multi-language support for domain descriptions (English only for v1)

## Appendix

### Glossary
- **Domain**: The type of business or system being simulated (e.g., flower shop, warehouse)
- **Agent**: An entity in the simulation that makes decisions (e.g., florist, customer)
- **Behavior**: The logic that determines what action an agent takes given their current context
- **LLM**: Large Language Model, the AI system used to generate behaviors

### Reference Links
- [Anthropic Claude API](https://docs.anthropic.com)
- [OpenAI API](https://platform.openai.com/docs)
- [Ollama](https://ollama.ai)
