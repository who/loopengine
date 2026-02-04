/**
 * LoopEngine Renderer Module
 *
 * Orchestrates all rendering with correct layer ordering and frame interpolation.
 * Ensures smooth 60fps animation by interpolating between server frames.
 *
 * Layer order (bottom to top):
 * 1. Label regions (soft clouds)
 * 2. Links (conduits)
 * 3. Particles (flowing dots)
 * 4. Agents (breathing shapes)
 * 5. Hover/selection overlays
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================

    const TARGET_FPS = 60;
    const FRAME_DURATION_MS = 1000 / TARGET_FPS;
    const MAX_INTERPOLATION_FRAMES = 5;  // Max frames to interpolate over

    // =========================================================================
    // Frame Interpolation State
    // =========================================================================

    let previousFrame = null;
    let currentFrame = null;
    let frameTimestamp = 0;
    let interpolationProgress = 0;

    // =========================================================================
    // Frame Interpolation Functions
    // =========================================================================

    /**
     * Linearly interpolate between two numbers.
     * @param {number} a - Start value
     * @param {number} b - End value
     * @param {number} t - Interpolation factor (0-1)
     * @returns {number} Interpolated value
     */
    function lerp(a, b, t) {
        return a + (b - a) * t;
    }

    /**
     * Interpolate a single agent between two frames.
     * @param {Object} prevAgent - Agent from previous frame
     * @param {Object} currAgent - Agent from current frame
     * @param {number} t - Interpolation factor (0-1)
     * @returns {Object} Interpolated agent
     */
    function interpolateAgent(prevAgent, currAgent, t) {
        if (!prevAgent) return currAgent;
        if (!currAgent) return prevAgent;

        return {
            ...currAgent,
            x: lerp(prevAgent.x, currAgent.x, t),
            y: lerp(prevAgent.y, currAgent.y, t),
            radius: lerp(prevAgent.radius || 20, currAgent.radius || 20, t),
            breathing_phase: lerp(prevAgent.breathing_phase || 0, currAgent.breathing_phase || 0, t),
            glow_intensity: lerp(prevAgent.glow_intensity || 0, currAgent.glow_intensity || 0, t)
        };
    }

    /**
     * Interpolate a single link between two frames.
     * @param {Object} prevLink - Link from previous frame
     * @param {Object} currLink - Link from current frame
     * @param {number} t - Interpolation factor (0-1)
     * @returns {Object} Interpolated link
     */
    function interpolateLink(prevLink, currLink, t) {
        if (!prevLink) return currLink;
        if (!currLink) return prevLink;

        // Interpolate control points
        const controlPoints = currLink.control_points.map((cp, i) => {
            const prevCp = prevLink.control_points[i];
            if (!prevCp) return cp;
            return [
                lerp(prevCp[0], cp[0], t),
                lerp(prevCp[1], cp[1], t)
            ];
        });

        return {
            ...currLink,
            control_points: controlPoints,
            thickness: lerp(prevLink.thickness || 2, currLink.thickness || 2, t),
            sway_phase: lerp(prevLink.sway_phase || 0, currLink.sway_phase || 0, t)
        };
    }

    /**
     * Interpolate a single particle between two frames.
     * @param {Object} prevParticle - Particle from previous frame
     * @param {Object} currParticle - Particle from current frame
     * @param {number} t - Interpolation factor (0-1)
     * @returns {Object} Interpolated particle
     */
    function interpolateParticle(prevParticle, currParticle, t) {
        if (!prevParticle) return currParticle;
        if (!currParticle) return prevParticle;

        return {
            ...currParticle,
            x: currParticle.x !== undefined ? lerp(prevParticle.x || currParticle.x, currParticle.x, t) : undefined,
            y: currParticle.y !== undefined ? lerp(prevParticle.y || currParticle.y, currParticle.y, t) : undefined,
            progress: lerp(prevParticle.progress || 0, currParticle.progress || 0, t),
            opacity: lerp(prevParticle.opacity !== undefined ? prevParticle.opacity : 1.0,
                         currParticle.opacity !== undefined ? currParticle.opacity : 1.0, t)
        };
    }

    /**
     * Interpolate a single label region between two frames.
     * @param {Object} prevRegion - Region from previous frame
     * @param {Object} currRegion - Region from current frame
     * @param {number} t - Interpolation factor (0-1)
     * @returns {Object} Interpolated region
     */
    function interpolateLabelRegion(prevRegion, currRegion, t) {
        if (!prevRegion) return currRegion;
        if (!currRegion) return prevRegion;

        // Interpolate hull points
        const hullPoints = currRegion.hull_points.map((hp, i) => {
            const prevHp = prevRegion.hull_points[i];
            if (!prevHp) return hp;
            return [
                lerp(prevHp[0], hp[0], t),
                lerp(prevHp[1], hp[1], t)
            ];
        });

        return {
            ...currRegion,
            hull_points: hullPoints,
            breathing_phase: lerp(prevRegion.breathing_phase || 0, currRegion.breathing_phase || 0, t)
        };
    }

    /**
     * Create a map from ID to object for fast lookup.
     * @param {Array} items - Array of objects with id property
     * @returns {Map} Map from id to object
     */
    function createIdMap(items) {
        const map = new Map();
        if (items) {
            for (const item of items) {
                map.set(item.id, item);
            }
        }
        return map;
    }

    /**
     * Interpolate between two frames.
     * @param {Object} prevFrame - Previous frame
     * @param {Object} currFrame - Current frame
     * @param {number} t - Interpolation factor (0-1)
     * @returns {Object} Interpolated frame
     */
    function interpolateFrame(prevFrame, currFrame, t) {
        if (!prevFrame) return currFrame;
        if (!currFrame) return prevFrame;
        if (t >= 1.0) return currFrame;
        if (t <= 0.0) return prevFrame;

        // Create lookup maps for previous frame
        const prevAgentMap = createIdMap(prevFrame.agents);
        const prevLinkMap = createIdMap(prevFrame.links);
        const prevParticleMap = createIdMap(prevFrame.particles);
        const prevRegionMap = new Map();
        if (prevFrame.label_regions) {
            for (const region of prevFrame.label_regions) {
                prevRegionMap.set(region.name, region);
            }
        }

        // Interpolate agents
        const agents = currFrame.agents.map(currAgent => {
            const prevAgent = prevAgentMap.get(currAgent.id);
            return interpolateAgent(prevAgent, currAgent, t);
        });

        // Interpolate links
        const links = currFrame.links.map(currLink => {
            const prevLink = prevLinkMap.get(currLink.id);
            return interpolateLink(prevLink, currLink, t);
        });

        // Interpolate particles
        const particles = currFrame.particles.map(currParticle => {
            const prevParticle = prevParticleMap.get(currParticle.id);
            return interpolateParticle(prevParticle, currParticle, t);
        });

        // Interpolate label regions
        let labelRegions = currFrame.label_regions;
        if (currFrame.label_regions && prevFrame.label_regions) {
            labelRegions = currFrame.label_regions.map(currRegion => {
                const prevRegion = prevRegionMap.get(currRegion.name);
                return interpolateLabelRegion(prevRegion, currRegion, t);
            });
        }

        return {
            tick: currFrame.tick,
            time: lerp(prevFrame.time, currFrame.time, t),
            agents: agents,
            links: links,
            particles: particles,
            label_regions: labelRegions
        };
    }

    // =========================================================================
    // Frame Management
    // =========================================================================

    /**
     * Push a new frame from the server.
     * @param {Object} frame - New frame from WebSocket
     */
    function pushFrame(frame) {
        previousFrame = currentFrame;
        currentFrame = frame;
        frameTimestamp = performance.now();
        interpolationProgress = 0;
    }

    /**
     * Get the interpolated frame for rendering.
     * @param {number} timestamp - Current render timestamp
     * @param {number} serverFrameRate - Expected server frame rate (e.g., 30)
     * @returns {Object} Frame to render (interpolated or current)
     */
    function getInterpolatedFrame(timestamp, serverFrameRate) {
        if (!currentFrame) return null;
        if (!previousFrame) return currentFrame;

        // Calculate how far we are between frames
        const serverFrameDuration = 1000 / serverFrameRate;
        const timeSinceFrame = timestamp - frameTimestamp;
        const t = Math.min(timeSinceFrame / serverFrameDuration, 1.0);

        return interpolateFrame(previousFrame, currentFrame, t);
    }

    // =========================================================================
    // Viewport Management
    // =========================================================================

    let viewport = {
        scale: 1.0,
        offsetX: 0,
        offsetY: 0
    };

    /**
     * Set the viewport transform.
     * @param {Object} newViewport - New viewport {scale, offsetX, offsetY}
     */
    function setViewport(newViewport) {
        viewport = { ...viewport, ...newViewport };
    }

    /**
     * Get the current viewport.
     * @returns {Object} Current viewport
     */
    function getViewport() {
        return viewport;
    }

    /**
     * Auto-center viewport on agents.
     * @param {Array} agents - Array of agents
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     */
    function autoCenterViewport(agents, canvasWidth, canvasHeight) {
        if (!agents || agents.length === 0) return;

        // Calculate bounding box of all agents
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const agent of agents) {
            minX = Math.min(minX, agent.x);
            minY = Math.min(minY, agent.y);
            maxX = Math.max(maxX, agent.x);
            maxY = Math.max(maxY, agent.y);
        }

        // Calculate content dimensions with padding
        const contentWidth = maxX - minX + 100;
        const contentHeight = maxY - minY + 100;

        // Scale to fit with padding
        viewport.scale = Math.min(
            (canvasWidth - 100) / contentWidth,
            (canvasHeight - 100) / contentHeight,
            2.0  // Max scale
        );
        viewport.scale = Math.max(viewport.scale, 0.25);  // Min scale

        // Center offset
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        viewport.offsetX = canvasWidth / 2 - centerX * viewport.scale;
        viewport.offsetY = canvasHeight / 2 - centerY * viewport.scale;
    }

    // =========================================================================
    // Main Rendering Functions
    // =========================================================================

    /**
     * Render a complete frame with all layers in correct order.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} frame - Frame to render (interpolated)
     * @param {number} animationTime - Current animation time
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     */
    function renderFrame(ctx, frame, animationTime, canvasWidth, canvasHeight) {
        if (!frame) return;

        // Update interaction module with current frame for hit testing
        if (typeof LoopEngineInteraction !== 'undefined') {
            LoopEngineInteraction.setFrame(frame);
        }

        // Layer 1: Label regions (lowest - soft translucent clouds)
        if (typeof LoopEngineLabels !== 'undefined' && frame.label_regions) {
            LoopEngineLabels.renderLabelRegions(ctx, frame.label_regions, animationTime, viewport);
        }

        // Layer 2: Links (conduits with sway)
        if (typeof LoopEngineLinks !== 'undefined' && frame.links) {
            LoopEngineLinks.renderLinks(ctx, frame.links, animationTime, viewport);
        }

        // Layer 3: Particles (small colored dots with trails)
        if (typeof LoopEngineParticles !== 'undefined' && frame.particles) {
            LoopEngineParticles.renderParticles(ctx, frame.particles, frame.links, animationTime, viewport);
        }

        // Layer 4: Agents (amoeba shapes with breathing pulse and inner glow)
        if (typeof LoopEngineAgents !== 'undefined' && frame.agents) {
            LoopEngineAgents.renderAgents(ctx, frame.agents, animationTime, viewport);
        }

        // Layer 5: Selection ring overlay (animated dashed circle)
        if (typeof LoopEngineInteraction !== 'undefined') {
            LoopEngineInteraction.renderSelectionRing(ctx, animationTime);
        }

        // Layer 6: Hover tooltips (on top of selection ring)
        if (typeof LoopEngineInteraction !== 'undefined') {
            LoopEngineInteraction.renderTooltip(ctx);
        }

        // Layer 7: Detail panel (slides in from right, on top of everything)
        if (typeof LoopEngineInteraction !== 'undefined') {
            LoopEngineInteraction.renderDetailPanel(ctx, animationTime);
        }
    }

    /**
     * Render status overlay (tick counter, stats, connection status).
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} frame - Current frame
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     * @param {boolean} connected - Connection status
     * @param {string} [connectionStatus] - Optional detailed status ('connecting', 'reconnecting', 'error')
     */
    function renderOverlay(ctx, frame, canvasWidth, canvasHeight, connected, connectionStatus) {
        if (!connected) {
            // Show different messages based on connection status
            const statusText = connectionStatus === 'reconnecting'
                ? 'Reconnecting to server...'
                : connectionStatus === 'error'
                ? 'Connection error. Retrying...'
                : 'Connecting to server...';

            const statusColor = connectionStatus === 'error' ? '#ff4444' : '#ff6b6b';

            ctx.fillStyle = statusColor;
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(statusText, canvasWidth / 2, canvasHeight / 2);

            // Show reconnection hint
            ctx.fillStyle = '#888888';
            ctx.font = '12px monospace';
            ctx.fillText('Check that the server is running on localhost:8000', canvasWidth / 2, canvasHeight / 2 + 25);
            return;
        }

        if (!frame || frame.tick < 0) {
            ctx.fillStyle = '#4a90d9';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for simulation data...', canvasWidth / 2, canvasHeight / 2);
            return;
        }

        // Draw tick/time info
        ctx.fillStyle = '#666666';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Tick: ' + frame.tick + ' | Time: ' + frame.time.toFixed(2) + 's', 10, 20);

        // Draw stats
        const agentCount = frame.agents ? frame.agents.length : 0;
        const linkCount = frame.links ? frame.links.length : 0;
        const particleCount = frame.particles ? frame.particles.length : 0;
        ctx.fillText(
            'Agents: ' + agentCount +
            ' | Links: ' + linkCount +
            ' | Particles: ' + particleCount,
            10, 36
        );
    }

    /**
     * Clear the canvas with background color.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     */
    function clearCanvas(ctx, width, height) {
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);
    }

    /**
     * Main render function - called every animation frame.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} timestamp - requestAnimationFrame timestamp
     * @param {number} animationTime - Accumulated animation time
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     * @param {boolean} connected - WebSocket connection status
     * @param {number} serverFrameRate - Server frame rate (default 30)
     */
    function render(ctx, timestamp, animationTime, canvasWidth, canvasHeight, connected, serverFrameRate) {
        serverFrameRate = serverFrameRate || 30;

        // Clear canvas
        clearCanvas(ctx, canvasWidth, canvasHeight);

        // Get interpolated frame
        const frame = getInterpolatedFrame(timestamp, serverFrameRate);

        // Auto-center viewport on first frame
        if (frame && viewport.offsetX === 0 && viewport.offsetY === 0 && frame.agents && frame.agents.length > 0) {
            autoCenterViewport(frame.agents, canvasWidth, canvasHeight);
        }

        // Render all layers
        if (frame) {
            renderFrame(ctx, frame, animationTime, canvasWidth, canvasHeight);
        }

        // Render overlay
        renderOverlay(ctx, frame, canvasWidth, canvasHeight, connected);

        // Render control bar (on top of everything)
        if (typeof LoopEngineControls !== 'undefined') {
            // Update tick counter from frame data
            if (frame && frame.tick !== undefined) {
                LoopEngineControls.updateTick(frame.tick);
            }
            LoopEngineControls.renderControlBar(ctx, canvasWidth, canvasHeight);
        }

        // Render corpus selector dropdown (in control bar area)
        if (typeof LoopEngineCorpus !== 'undefined') {
            LoopEngineCorpus.renderCorpusSelector(ctx, canvasWidth, canvasHeight);
        }

        // Render GA panel (top-right corner, on top of control bar)
        if (typeof LoopEngineGA !== 'undefined') {
            LoopEngineGA.renderGAPanel(ctx, canvasWidth, canvasHeight);
        }

        // Render Discovery panel (top-left corner, below status)
        if (typeof LoopEngineDiscovery !== 'undefined') {
            LoopEngineDiscovery.renderDiscoveryPanel(ctx, canvasWidth, canvasHeight);
        }
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineRenderer = {
        // Frame management
        pushFrame: pushFrame,
        getInterpolatedFrame: getInterpolatedFrame,

        // Viewport
        setViewport: setViewport,
        getViewport: getViewport,
        autoCenterViewport: autoCenterViewport,

        // Rendering
        render: render,
        renderFrame: renderFrame,
        renderOverlay: renderOverlay,
        clearCanvas: clearCanvas,

        // Interpolation utilities
        interpolateFrame: interpolateFrame,
        lerp: lerp,

        // Constants
        TARGET_FPS: TARGET_FPS
    };

})(typeof window !== 'undefined' ? window : this);
