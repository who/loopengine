# GA Experiment Results

## Experiment Configuration

- **Generations**: 100
- **Population Size**: 50
- **Selection Count**: 10 (top 20%)
- **Mutation Rate**: 0.1
- **Mutation Magnitude**: 0.05
- **Ticks per Evaluation**: 1000
- **Selection Type**: Rank
- **Crossover Type**: Uniform

## Results Summary

### Tom (sandwich_maker)

**Best Genome:**
| Trait | Value | Interpretation |
|-------|-------|----------------|
| speed | 0.152 | LOW |
| consistency | 0.626 | NEUTRAL |
| ingredient_intuition | 0.933 | HIGH |
| stress_tolerance | 0.048 | LOW |
| waste_minimization | 0.242 | LOW |

**Fitness**: 0.0 (best achievable due to simulation bug)

**Observations:**
- Average fitness across population: ~-1.5 (negative due to queue depth penalty)
- Fitness variance: -3.5 to 0.0 range
- No meaningful convergence observed - random drift around max fitness of 0.0

### Alex (cashier)

**Best Genome:**
| Trait | Value | Interpretation |
|-------|-------|----------------|
| speed | 0.450 | NEUTRAL |
| accuracy | 0.248 | LOW |
| friendliness | 0.980 | HIGH |
| stress_tolerance | 0.335 | NEUTRAL |
| upselling | 0.295 | LOW |

**Fitness**: 0.0

**Observations:**
- Converged quickly (minimal variance in last 10 generations)
- Alex fitness is 0.0 because customers_served = 0 (no sandwiches returned from Tom)

### Maria (owner)

**Best Genome:**
| Trait | Value | Interpretation |
|-------|-------|----------------|
| supply_forecasting | 0.140 | LOW |
| observation | 0.345 | NEUTRAL |
| decisiveness | 0.497 | NEUTRAL |
| delegation | 0.635 | NEUTRAL |
| cost_sensitivity | 0.966 | HIGH |

**Fitness**: 0.0

**Observations:**
- Converged quickly
- Maria fitness depends on Tom's sandwich throughput (which is 0)

## Critical Finding: Simulation Bug

The experiments revealed a **critical bug in the OODA loop implementation** that prevents meaningful fitness evaluation:

### Bug Description

In `src/loopengine/engine/loop.py`, the `_do_sense()` function runs on **every tick** during the SENSE phase, not just the first tick. This causes:

1. First SENSE tick: `input_buffer` → `sensed_inputs`, buffer cleared
2. Second SENSE tick: empty `input_buffer` → `sensed_inputs` (overwrites!)
3. By DECIDE phase: `sensed_inputs` is empty

### Impact

- Agents receive inputs in their buffer between OODA cycles
- When SENSE phase starts, inputs are read and buffer is cleared
- Subsequent SENSE ticks overwrite with empty buffer
- Policies receive empty sensed_inputs, produce no outputs
- **Result**: No particle flow through the system after first sense

### Evidence

Traced simulation output shows:
- Tick 87: Tom SENSE(tick 3), sensed=1 ← input received
- Tick 88: Tom SENSE(tick 4), sensed=0 ← overwritten!
- Tick 98: Tom DECIDE(tick 0), sensed=0 ← policy gets nothing

### Workaround

The phase behavior functions should only execute on `phase_tick == 0`:

```python
def _do_sense(agent: Agent) -> None:
    if agent.phase_tick == 0:  # Only sense at start of phase
        agent.internal_state["sensed_inputs"] = list(agent.input_buffer)
        agent.input_buffer.clear()
```

## Fitness Analysis (Under Bug Conditions)

Despite the bug, some fitness signal exists:

### Tom Fitness Components
- **Throughput**: sandwiches_completed / ticks = 0/1000 = 0
- **Quality**: avg_consistency = 0 (no sandwiches)
- **Waste ratio**: 0 (no waste, no ingredients)
- **Queue penalty**: max_queue_depth * 0.1 = 15-50 * 0.1 = -1.5 to -5.0

**Net fitness**: 0 - 0 - queue_penalty = -1.5 to -5.0

The only fitness signal is the queue depth penalty (Alex accumulates waiting customers). This explains why fitness varies between -5.0 and 0.0.

### What the GA Is Actually Optimizing

Since throughput = 0 for all genomes, the GA is selecting for genomes that **minimize queue depth penalty**. Queue depth is affected by:
- Customer arrival rate (external, not controllable)
- Timing of OODA cycles (affected by loop_period, not genome)

The genome traits have **no actual effect** under current conditions.

## Recommendations

1. **Fix the OODA loop bug** - Add phase_tick == 0 guards to phase behavior functions
2. **Re-run experiments** after fix to get meaningful results
3. **Consider shorter loop periods** - Current periods (20-30 ticks) may be too slow for the particle flow rates

## Files Generated

- `experiments/ga_experiments.py` - Experiment script
- `experiments/results/sandwich_maker_evolution.json` - Tom's 100-gen evolution data
- `experiments/results/cashier_evolution.json` - Alex's 100-gen evolution data
- `experiments/results/owner_evolution.json` - Maria's 100-gen evolution data
