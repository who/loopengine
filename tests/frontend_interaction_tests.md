# Frontend Interaction Integration Tests

Manual test checklist for LoopEngine canvas interactions.

## Prerequisites

1. Start the development server: `uv run flask --app app.main run --port 8000`
2. Open `http://localhost:8000` in a browser
3. Wait for WebSocket connection and agent rendering

---

## 1. Hover Behavior Tests

### 1.1 Basic Hover Detection
- [ ] Hover over an agent - cursor changes from `grab` to `pointer`
- [ ] Hover over empty canvas space - cursor shows `grab`
- [ ] Move mouse quickly across agents - hover state updates without lag

### 1.2 Tooltip Display
- [ ] Hover over agent - tooltip appears near agent (right side if space permits)
- [ ] Tooltip shows agent name and role
- [ ] Tooltip shows OODA phase with color indicator (teal/yellow/red/green)
- [ ] Tooltip shows input buffer depth
- [ ] Tooltip displays genome traits as bar chart with values

### 1.3 Tooltip Positioning
- [ ] Tooltip near right edge flips to left side of agent
- [ ] Tooltip near bottom edge adjusts upward
- [ ] Tooltip near top edge adjusts downward
- [ ] Tooltip remains fully visible at all canvas edges

### 1.4 Hover Visual Feedback
- [ ] Hovered agent has increased glow effect (+50%)
- [ ] Hovered agent radius scales up (115%)
- [ ] Connected links brighten when hovering an agent

### 1.5 Hover State Cleanup
- [ ] Moving mouse off agent - tooltip disappears
- [ ] Mouse leaving canvas - hover state clears
- [ ] Hovering over control bar - no agent tooltip shows

---

## 2. Click Selection Tests

### 2.1 Basic Selection
- [ ] Click on agent - agent becomes selected
- [ ] Selected agent shows animated dashed ring
- [ ] Selection ring pulses (expands/contracts)
- [ ] Selection ring dashes animate (rotating)
- [ ] Click on empty canvas - selection clears

### 2.2 Detail Panel
- [ ] Selecting agent - detail panel slides in from right
- [ ] Panel shows agent name (bold, top)
- [ ] Panel shows agent role
- [ ] Panel shows OODA phase with colored label
- [ ] Panel shows buffer depth
- [ ] Panel shows labels list (if any)
- [ ] Panel shows genome traits with bar chart
- [ ] Panel shows connected links with direction arrows (→/←)
- [ ] Link type shown in parentheses with appropriate color

### 2.3 Selection Animation
- [ ] View pans smoothly to center selected agent
- [ ] Pan accounts for panel width (agent not hidden by panel)
- [ ] Pan uses ease-out animation (smooth deceleration)
- [ ] Multiple rapid selections - each triggers new pan animation

### 2.4 Selection State Management
- [ ] Clicking different agent - selection changes to new agent
- [ ] Clicking same agent - no state change (remains selected)
- [ ] Selecting then hovering another agent - both effects visible
- [ ] Combined glow effect for hover + selection

### 2.5 Detail Panel Animation
- [ ] Panel slides in smoothly when selecting
- [ ] Panel slides out smoothly when deselecting
- [ ] Rapid select/deselect - animation transitions smoothly

---

## 3. Zoom and Pan Tests

### 3.1 Mouse Wheel Zoom
- [ ] Scroll up - zoom in (scale increases)
- [ ] Scroll down - zoom out (scale decreases)
- [ ] Zoom centers on cursor position (world point under cursor stays fixed)
- [ ] Zoom transitions smoothly with animation
- [ ] Minimum zoom limit: 0.1 (macro view)
- [ ] Maximum zoom limit: 5.0 (micro view)

### 3.2 Click-Drag Pan
- [ ] Click and drag on empty space - canvas pans
- [ ] Cursor changes to `grabbing` during drag
- [ ] Pan is smooth and immediate (1:1 with mouse movement)
- [ ] Clicking on agent does NOT initiate pan (only selection)
- [ ] Release drag - cursor returns to `grab`

### 3.3 Pan Momentum (Inertia)
- [ ] Quick drag and release - canvas continues moving with momentum
- [ ] Momentum gradually decreases (friction: 0.92)
- [ ] Momentum stops when velocity is very low
- [ ] Starting new drag - momentum stops immediately

### 3.4 Drag-Click Suppression
- [ ] Drag then release - click event is suppressed
- [ ] Small movement (<5px) - click event fires normally
- [ ] Threshold prevents accidental selections during pan

### 3.5 Touch Gestures (Mobile/Tablet)
- [ ] Single touch drag - pans viewport
- [ ] Two-finger pinch - zooms in/out
- [ ] Pinch zoom centers on pinch midpoint
- [ ] Pan momentum works with touch
- [ ] Touch on agent - no pan initiated

### 3.6 Zoom Controls
- [ ] Zoom respects MIN_ZOOM (0.1) and MAX_ZOOM (5.0) limits
- [ ] Zoom near limits - clamping works correctly
- [ ] Zoom level accessible via `LoopEngineInteraction.getZoomLevel()`
- [ ] `setZoomLevel()` with animate=true - smooth transition
- [ ] `setZoomLevel()` with animate=false - immediate change
- [ ] `zoomToFit()` - fits all agents in view

---

## 4. Time Controls Tests

### 4.1 Play/Pause Button
- [ ] Initial state - shows pause icon (two vertical bars), simulation running
- [ ] Click button - toggles to play icon (triangle), simulation pauses
- [ ] Click again - toggles back to pause icon, simulation resumes
- [ ] Hover over button - color brightens (hover effect)

### 4.2 Speed Slider
- [ ] Slider positioned after play/pause button
- [ ] Slider shows current speed position
- [ ] Drag knob - speed changes in real-time
- [ ] Speed uses logarithmic scale (more precision at low speeds)
- [ ] Minimum speed: 0.25x
- [ ] Maximum speed: 10.0x
- [ ] Default speed: 1.0x
- [ ] Click on track - knob jumps to position
- [ ] Speed label shows current value (e.g., "1.00x")

### 4.3 Speed Slider Visual Feedback
- [ ] Filled track portion matches speed position
- [ ] Knob has border when hovering/dragging
- [ ] Slider works at all zoom levels

### 4.4 Tick Counter
- [ ] Counter shows "Tick: N" at right side of control bar
- [ ] Counter updates as simulation progresses
- [ ] Counter reflects actual simulation tick from server

### 4.5 Control Bar Interaction Isolation
- [ ] Mouse wheel in control bar - does NOT zoom canvas
- [ ] Click in control bar - does NOT deselect agents
- [ ] Hover in control bar - does NOT show agent tooltips
- [ ] Drag in control bar - does NOT pan canvas

---

## 5. Edge Cases

### 5.1 Rapid Interaction Sequences
- [ ] Rapid clicks on multiple agents - selection updates correctly each time
- [ ] Rapid hover movements - no visual glitches or stuck states
- [ ] Rapid zoom in/out - animation queues correctly or latest wins
- [ ] Zoom while panning - both interactions work simultaneously

### 5.2 Zoom Limits
- [ ] At MIN_ZOOM (0.1) - further zoom out has no effect
- [ ] At MAX_ZOOM (5.0) - further zoom in has no effect
- [ ] Zoom at limit - no jitter or oscillation

### 5.3 Pan Limits
- [ ] Pan very far - agents can be moved off-screen
- [ ] No hard boundaries on pan (intentional - allows exploration)
- [ ] Very far pan + zoom fit - agents return to view

### 5.4 Empty/Loading States
- [ ] No agents loaded - hover/click has no effect
- [ ] Disconnected from server - "Connecting..." message shown
- [ ] Waiting for data - "Waiting for simulation data..." message shown

### 5.5 Selection During Animation
- [ ] Select agent during pan momentum - momentum stops
- [ ] Select agent during zoom animation - new pan animation starts
- [ ] Deselect during panel slide-in - panel slides back out

### 5.6 Resize and Viewport Changes
- [ ] Browser window resize - canvas adjusts, interactions work
- [ ] Control bar stays at bottom after resize
- [ ] Detail panel stays at right after resize

### 5.7 Multi-Touch Edge Cases
- [ ] Single touch on agent - selects (not pan)
- [ ] Two fingers then lift one - no crash, state recovers
- [ ] Touch outside canvas bounds - no crash

### 5.8 Concurrent State
- [ ] Hover agent A, select agent B - both states coexist
- [ ] Hover + select same agent - combined glow effect (capped at 1.0)
- [ ] Select, then hover another agent - links from hovered agent brighten

---

## 6. Performance Tests

### 6.1 Frame Rate
- [ ] Hover state changes - no FPS drop
- [ ] Selection animation - smooth 60fps
- [ ] Zoom animation - smooth 60fps
- [ ] Pan with many agents - responsive

### 6.2 Memory
- [ ] Long session with many interactions - no memory leak
- [ ] Rapid select/deselect - stable memory usage

---

## Test Execution Log

| Date | Tester | Browser | Version | Pass/Fail | Notes |
|------|--------|---------|---------|-----------|-------|
|      |        |         |         |           |       |

## Known Issues

(Document any discovered issues here with links to bug reports)

---

## Automated Test Considerations

For future automation with Playwright/Puppeteer:

1. **Hover tests**: `page.hover()` at calculated agent positions
2. **Click tests**: `page.click()` with coordinate targeting
3. **Drag tests**: `page.mouse.down()`, `move()`, `up()` sequences
4. **Wheel tests**: `page.mouse.wheel()` for zoom
5. **Touch tests**: `page.touchscreen.tap()` and gesture simulation
6. **Assertions**: Canvas snapshots, DOM state checks, WebSocket message verification

Canvas-based testing requires:
- Agent position extraction via `LoopEngineInteraction.hitTestAgents()` or frame data
- State inspection via exported module functions
- Screenshot comparison for visual regression
