/**
 * LoopEngine Interaction Module
 *
 * Handles mouse hover detection, tooltips, and link highlighting for agents.
 * Provides hover tooltips showing agent details including genome traits bar chart.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // State
    // =========================================================================

    let hoveredAgent = null;
    let mouseX = 0;
    let mouseY = 0;
    let canvas = null;
    let currentFrame = null;

    // Hover highlight configuration
    const HOVER_GLOW_INCREASE = 0.5;  // Added glow when hovering
    const HOVER_RADIUS_SCALE = 1.15;  // Scale factor for hovered agent

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * Initialize interaction module with canvas element.
     * @param {HTMLCanvasElement} canvasElement - The canvas element
     */
    function init(canvasElement) {
        canvas = canvasElement;

        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseleave', handleMouseLeave);
    }

    /**
     * Update current frame data for hit testing.
     * @param {Object} frame - Current frame from server/interpolation
     */
    function setFrame(frame) {
        currentFrame = frame;
    }

    // =========================================================================
    // Event Handlers
    // =========================================================================

    /**
     * Handle mouse move events.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseMove(event) {
        const rect = canvas.getBoundingClientRect();
        mouseX = event.clientX - rect.left;
        mouseY = event.clientY - rect.top;

        // Perform hit test against agents
        hoveredAgent = hitTestAgents(mouseX, mouseY);
    }

    /**
     * Handle mouse leave events.
     */
    function handleMouseLeave() {
        hoveredAgent = null;
    }

    // =========================================================================
    // Hit Testing
    // =========================================================================

    /**
     * Test if mouse position hits any agent.
     * @param {number} x - Mouse X in canvas coordinates
     * @param {number} y - Mouse Y in canvas coordinates
     * @returns {Object|null} The agent hit, or null
     */
    function hitTestAgents(x, y) {
        if (!currentFrame || !currentFrame.agents) {
            return null;
        }

        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        // Test agents in reverse order (top to bottom in render order)
        const agents = currentFrame.agents;
        for (let i = agents.length - 1; i >= 0; i--) {
            const agent = agents[i];

            // Transform agent position to screen space
            const screenX = agent.x * viewport.scale + viewport.offsetX;
            const screenY = agent.y * viewport.scale + viewport.offsetY;
            const screenRadius = (agent.radius || 20) * viewport.scale;

            // Distance from mouse to agent center
            const dx = x - screenX;
            const dy = y - screenY;
            const distSq = dx * dx + dy * dy;

            // Hit if within radius (with some tolerance)
            if (distSq <= screenRadius * screenRadius * 1.2) {
                return agent;
            }
        }

        return null;
    }

    // =========================================================================
    // Tooltip Rendering
    // =========================================================================

    /**
     * Render hover tooltip near hovered agent.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    function renderTooltip(ctx) {
        if (!hoveredAgent) return;

        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        // Calculate tooltip position
        const agentScreenX = hoveredAgent.x * viewport.scale + viewport.offsetX;
        const agentScreenY = hoveredAgent.y * viewport.scale + viewport.offsetY;
        const agentScreenRadius = (hoveredAgent.radius || 20) * viewport.scale;

        // Position tooltip to the right of agent
        let tooltipX = agentScreenX + agentScreenRadius + 15;
        let tooltipY = agentScreenY - 20;

        // Calculate tooltip dimensions
        const padding = 10;
        const lineHeight = 16;
        const barHeight = 10;
        const barWidth = 80;
        const labelWidth = 90;

        // Count genome traits for height calculation
        const genomeTraits = hoveredAgent.genome || {};
        const traitCount = Object.keys(genomeTraits).length;

        // Tooltip dimensions
        const tooltipWidth = labelWidth + barWidth + padding * 3;
        const headerHeight = lineHeight * 3 + padding;  // Name+role, OODA phase, buffer depth
        const genomeHeight = traitCount > 0 ? (barHeight + 4) * traitCount + lineHeight + padding : 0;
        const tooltipHeight = headerHeight + genomeHeight + padding * 2;

        // Adjust position if tooltip would go off-screen
        const canvasWidth = ctx.canvas._logicalWidth || ctx.canvas.width;
        const canvasHeight = ctx.canvas._logicalHeight || ctx.canvas.height;

        if (tooltipX + tooltipWidth > canvasWidth - 10) {
            tooltipX = agentScreenX - agentScreenRadius - tooltipWidth - 15;
        }
        if (tooltipY + tooltipHeight > canvasHeight - 10) {
            tooltipY = canvasHeight - tooltipHeight - 10;
        }
        if (tooltipY < 10) {
            tooltipY = 10;
        }

        ctx.save();

        // Draw tooltip background
        ctx.fillStyle = 'rgba(30, 30, 45, 0.95)';
        ctx.strokeStyle = 'rgba(100, 100, 140, 0.8)';
        ctx.lineWidth = 1;

        // Rounded rectangle
        drawRoundedRect(ctx, tooltipX, tooltipY, tooltipWidth, tooltipHeight, 6);
        ctx.fill();
        ctx.stroke();

        // Draw content
        let yPos = tooltipY + padding;

        // Agent name and role
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(hoveredAgent.name + ' - ' + hoveredAgent.role, tooltipX + padding, yPos);
        yPos += lineHeight + 4;

        // OODA phase
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#aaaaaa';
        ctx.fillText('OODA: ', tooltipX + padding, yPos);
        ctx.fillStyle = getOodaPhaseColor(hoveredAgent.ooda_phase);
        ctx.fillText(hoveredAgent.ooda_phase.toUpperCase(), tooltipX + padding + 45, yPos);
        yPos += lineHeight;

        // Input buffer depth
        ctx.fillStyle = '#aaaaaa';
        const bufferDepth = hoveredAgent.input_buffer_depth !== undefined
            ? hoveredAgent.input_buffer_depth
            : 0;
        ctx.fillText('Buffer: ' + bufferDepth + ' items', tooltipX + padding, yPos);
        yPos += lineHeight + 8;

        // Genome traits bar chart
        if (traitCount > 0) {
            ctx.fillStyle = '#888888';
            ctx.font = '11px sans-serif';
            ctx.fillText('Genome Traits:', tooltipX + padding, yPos);
            yPos += lineHeight;

            for (const [trait, value] of Object.entries(genomeTraits)) {
                // Trait name (truncate if needed)
                ctx.fillStyle = '#cccccc';
                ctx.font = '10px sans-serif';
                let traitName = trait.replace(/_/g, ' ');
                if (traitName.length > 12) {
                    traitName = traitName.substring(0, 11) + '...';
                }
                ctx.fillText(traitName, tooltipX + padding, yPos + 1);

                // Bar background
                const barX = tooltipX + labelWidth;
                ctx.fillStyle = 'rgba(60, 60, 80, 0.8)';
                ctx.fillRect(barX, yPos, barWidth, barHeight);

                // Bar fill
                const fillWidth = Math.max(0, Math.min(1, value)) * barWidth;
                ctx.fillStyle = getTraitBarColor(value);
                ctx.fillRect(barX, yPos, fillWidth, barHeight);

                // Value text
                ctx.fillStyle = '#ffffff';
                ctx.font = '9px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(value.toFixed(2), barX + barWidth - 2, yPos + 1);
                ctx.textAlign = 'left';

                yPos += barHeight + 4;
            }
        }

        ctx.restore();
    }

    /**
     * Draw a rounded rectangle path.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Width
     * @param {number} height - Height
     * @param {number} radius - Corner radius
     */
    function drawRoundedRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.arcTo(x + width, y, x + width, y + radius, radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.arcTo(x + width, y + height, x + width - radius, y + height, radius);
        ctx.lineTo(x + radius, y + height);
        ctx.arcTo(x, y + height, x, y + height - radius, radius);
        ctx.lineTo(x, y + radius);
        ctx.arcTo(x, y, x + radius, y, radius);
        ctx.closePath();
    }

    /**
     * Get color for OODA phase indicator.
     * @param {string} phase - OODA phase name
     * @returns {string} Color string
     */
    function getOodaPhaseColor(phase) {
        switch (phase.toLowerCase()) {
            case 'sense': return '#4ecdc4';   // Teal
            case 'orient': return '#f9ca24';  // Yellow
            case 'decide': return '#ff6b6b';  // Red
            case 'act': return '#2ecc71';     // Green
            default: return '#888888';
        }
    }

    /**
     * Get color for trait bar based on value.
     * @param {number} value - Trait value (0-1)
     * @returns {string} Color string
     */
    function getTraitBarColor(value) {
        // Gradient from blue (low) to green (medium) to orange (high)
        if (value < 0.33) {
            return '#3498db';  // Blue
        } else if (value < 0.66) {
            return '#2ecc71';  // Green
        } else {
            return '#f39c12';  // Orange
        }
    }

    // =========================================================================
    // Hover Highlight Effects
    // =========================================================================

    /**
     * Get hover state for an agent (glow and scale adjustments).
     * @param {Object} agent - Agent to check
     * @returns {Object} Hover state {glowBoost, scaleBoost, isHovered}
     */
    function getAgentHoverState(agent) {
        if (!hoveredAgent || agent.id !== hoveredAgent.id) {
            return { glowBoost: 0, scaleBoost: 1.0, isHovered: false };
        }
        return {
            glowBoost: HOVER_GLOW_INCREASE,
            scaleBoost: HOVER_RADIUS_SCALE,
            isHovered: true
        };
    }

    /**
     * Check if a link should be highlighted (connected to hovered agent).
     * @param {Object} link - Link to check
     * @returns {boolean} True if link should be highlighted
     */
    function isLinkHighlighted(link) {
        if (!hoveredAgent) return false;
        return link.source_id === hoveredAgent.id || link.dest_id === hoveredAgent.id;
    }

    /**
     * Get the currently hovered agent.
     * @returns {Object|null} The hovered agent or null
     */
    function getHoveredAgent() {
        return hoveredAgent;
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineInteraction = {
        init: init,
        setFrame: setFrame,
        renderTooltip: renderTooltip,
        getAgentHoverState: getAgentHoverState,
        isLinkHighlighted: isLinkHighlighted,
        getHoveredAgent: getHoveredAgent,
        hitTestAgents: hitTestAgents
    };

})(typeof window !== 'undefined' ? window : this);
