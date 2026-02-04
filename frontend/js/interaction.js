/**
 * LoopEngine Interaction Module
 *
 * Handles mouse hover detection, tooltips, click selection, and link highlighting.
 * Provides hover tooltips and a slide-in detail panel for selected agents.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // State
    // =========================================================================

    let hoveredAgent = null;
    let selectedAgent = null;
    let mouseX = 0;
    let mouseY = 0;
    let canvas = null;
    let currentFrame = null;

    // Hover highlight configuration
    const HOVER_GLOW_INCREASE = 0.5;  // Added glow when hovering
    const HOVER_RADIUS_SCALE = 1.15;  // Scale factor for hovered agent

    // Selection configuration
    const SELECTION_GLOW_INCREASE = 0.7;  // Added glow when selected
    const SELECTION_RING_WIDTH = 3;       // Selection ring stroke width
    const SELECTION_RING_COLOR = 'rgba(255, 255, 255, 0.9)';
    const SELECTION_RING_PULSE_SPEED = 2.0;  // Pulse frequency

    // Detail panel configuration
    const PANEL_WIDTH = 280;
    const PANEL_SLIDE_SPEED = 0.15;  // Animation speed (0-1 per frame)
    let panelSlideProgress = 0;      // 0 = hidden, 1 = fully visible

    // Pan animation for centering on selection
    let panAnimation = null;  // {startX, startY, targetX, targetY, progress}

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
        canvas.addEventListener('click', handleClick);
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

    /**
     * Handle click events for agent selection.
     * @param {MouseEvent} event - Mouse event
     */
    function handleClick(event) {
        const rect = canvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;

        // Perform hit test against agents
        const clickedAgent = hitTestAgents(clickX, clickY);

        if (clickedAgent) {
            // Select the clicked agent
            selectAgent(clickedAgent);
        } else {
            // Click on empty space - deselect
            deselectAgent();
        }
    }

    /**
     * Select an agent and initiate pan animation to center it.
     * @param {Object} agent - Agent to select
     */
    function selectAgent(agent) {
        selectedAgent = agent;
        panelSlideProgress = 0;  // Start panel slide animation

        // Initiate pan animation to center the agent
        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        const canvasWidth = canvas._logicalWidth || canvas.width;
        const canvasHeight = canvas._logicalHeight || canvas.height;

        // Target offset to center the agent (accounting for detail panel)
        const panelOffset = PANEL_WIDTH / 2;  // Shift center left to account for panel
        const targetOffsetX = (canvasWidth / 2 - panelOffset) - agent.x * viewport.scale;
        const targetOffsetY = canvasHeight / 2 - agent.y * viewport.scale;

        panAnimation = {
            startX: viewport.offsetX,
            startY: viewport.offsetY,
            targetX: targetOffsetX,
            targetY: targetOffsetY,
            progress: 0
        };
    }

    /**
     * Deselect the currently selected agent.
     */
    function deselectAgent() {
        selectedAgent = null;
        panAnimation = null;
    }

    /**
     * Update animations (called each frame).
     * @param {number} deltaTime - Time since last frame in seconds
     */
    function updateAnimations(deltaTime) {
        // Update panel slide animation
        if (selectedAgent && panelSlideProgress < 1) {
            panelSlideProgress = Math.min(1, panelSlideProgress + PANEL_SLIDE_SPEED);
        } else if (!selectedAgent && panelSlideProgress > 0) {
            panelSlideProgress = Math.max(0, panelSlideProgress - PANEL_SLIDE_SPEED);
        }

        // Update pan animation
        if (panAnimation && panAnimation.progress < 1) {
            panAnimation.progress = Math.min(1, panAnimation.progress + 0.05);

            // Ease-out interpolation
            const t = easeOutCubic(panAnimation.progress);
            const newOffsetX = lerp(panAnimation.startX, panAnimation.targetX, t);
            const newOffsetY = lerp(panAnimation.startY, panAnimation.targetY, t);

            if (typeof LoopEngineRenderer !== 'undefined') {
                LoopEngineRenderer.setViewport({
                    offsetX: newOffsetX,
                    offsetY: newOffsetY
                });
            }

            // Clear animation when complete
            if (panAnimation.progress >= 1) {
                panAnimation = null;
            }
        }
    }

    /**
     * Ease-out cubic function for smooth animation.
     * @param {number} t - Input (0-1)
     * @returns {number} Eased value (0-1)
     */
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    /**
     * Linear interpolation.
     * @param {number} a - Start value
     * @param {number} b - End value
     * @param {number} t - Interpolation factor (0-1)
     * @returns {number} Interpolated value
     */
    function lerp(a, b, t) {
        return a + (b - a) * t;
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
    // Hover and Selection Highlight Effects
    // =========================================================================

    /**
     * Get hover/selection state for an agent (glow and scale adjustments).
     * @param {Object} agent - Agent to check
     * @returns {Object} State {glowBoost, scaleBoost, isHovered, isSelected}
     */
    function getAgentHoverState(agent) {
        const isHovered = hoveredAgent && agent.id === hoveredAgent.id;
        const isSelected = selectedAgent && agent.id === selectedAgent.id;

        if (!isHovered && !isSelected) {
            return { glowBoost: 0, scaleBoost: 1.0, isHovered: false, isSelected: false };
        }

        // Selection takes priority for glow, but hover adds to it
        let glowBoost = 0;
        let scaleBoost = 1.0;

        if (isSelected) {
            glowBoost += SELECTION_GLOW_INCREASE;
        }
        if (isHovered) {
            glowBoost += HOVER_GLOW_INCREASE;
            scaleBoost = HOVER_RADIUS_SCALE;
        }

        return {
            glowBoost: Math.min(1.0, glowBoost),
            scaleBoost: scaleBoost,
            isHovered: isHovered,
            isSelected: isSelected
        };
    }

    /**
     * Render selection ring around selected agent.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} animationTime - Current animation time
     */
    function renderSelectionRing(ctx, animationTime) {
        if (!selectedAgent) return;

        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        const screenX = selectedAgent.x * viewport.scale + viewport.offsetX;
        const screenY = selectedAgent.y * viewport.scale + viewport.offsetY;
        const baseRadius = (selectedAgent.radius || 20) * viewport.scale;

        // Pulsing ring radius
        const pulsePhase = animationTime * SELECTION_RING_PULSE_SPEED;
        const pulseAmount = 1 + 0.08 * Math.sin(pulsePhase * Math.PI * 2);
        const ringRadius = baseRadius * 1.3 * pulseAmount;

        ctx.save();

        // Draw selection ring
        ctx.strokeStyle = SELECTION_RING_COLOR;
        ctx.lineWidth = SELECTION_RING_WIDTH;
        ctx.setLineDash([8, 4]);  // Dashed line for selection
        ctx.lineDashOffset = -animationTime * 30;  // Animated dash

        ctx.beginPath();
        ctx.arc(screenX, screenY, ringRadius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.restore();
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

    /**
     * Get the currently selected agent.
     * @returns {Object|null} The selected agent or null
     */
    function getSelectedAgent() {
        return selectedAgent;
    }

    // =========================================================================
    // Detail Panel Rendering
    // =========================================================================

    /**
     * Render the slide-in detail panel for the selected agent.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} animationTime - Current animation time
     */
    function renderDetailPanel(ctx, animationTime) {
        // Update animations
        updateAnimations(0.016);  // Assume ~60fps

        // Don't render if panel is fully hidden
        if (panelSlideProgress <= 0 && !selectedAgent) return;

        const canvasWidth = ctx.canvas._logicalWidth || ctx.canvas.width;
        const canvasHeight = ctx.canvas._logicalHeight || ctx.canvas.height;

        // Calculate panel position with slide animation (ease-out)
        const slideEase = easeOutCubic(panelSlideProgress);
        const panelX = canvasWidth - (PANEL_WIDTH * slideEase);

        // Get the agent to display (use selectedAgent or last known for slide-out)
        const agent = selectedAgent;
        if (!agent && panelSlideProgress <= 0) return;

        ctx.save();

        // Draw panel background
        ctx.fillStyle = 'rgba(25, 25, 40, 0.95)';
        ctx.strokeStyle = 'rgba(80, 80, 120, 0.6)';
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.rect(panelX, 0, PANEL_WIDTH, canvasHeight);
        ctx.fill();
        ctx.stroke();

        // Don't render content if panel is sliding out and we have no agent
        if (!agent) {
            ctx.restore();
            return;
        }

        const padding = 16;
        const lineHeight = 20;
        const sectionGap = 16;
        let yPos = padding;

        // =====================================================================
        // Header: Agent name and role
        // =====================================================================
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(agent.name, panelX + padding, yPos);
        yPos += lineHeight;

        ctx.fillStyle = '#888888';
        ctx.font = '13px sans-serif';
        ctx.fillText(agent.role.replace(/_/g, ' '), panelX + padding, yPos);
        yPos += lineHeight + sectionGap;

        // =====================================================================
        // OODA Phase and Buffer
        // =====================================================================
        ctx.fillStyle = '#aaaaaa';
        ctx.font = '12px sans-serif';
        ctx.fillText('OODA Phase:', panelX + padding, yPos);
        ctx.fillStyle = getOodaPhaseColor(agent.ooda_phase);
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText(agent.ooda_phase.toUpperCase(), panelX + padding + 85, yPos);
        yPos += lineHeight;

        ctx.fillStyle = '#aaaaaa';
        ctx.font = '12px sans-serif';
        const bufferDepth = agent.input_buffer_depth !== undefined ? agent.input_buffer_depth : 0;
        ctx.fillText('Buffer Depth: ' + bufferDepth, panelX + padding, yPos);
        yPos += lineHeight + sectionGap;

        // =====================================================================
        // Labels Section
        // =====================================================================
        const labels = agent.labels || [];
        if (labels.length > 0) {
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px sans-serif';
            ctx.fillText('Labels', panelX + padding, yPos);
            yPos += lineHeight;

            ctx.fillStyle = '#cccccc';
            ctx.font = '12px sans-serif';
            for (const label of labels) {
                ctx.fillText('• ' + label, panelX + padding + 8, yPos);
                yPos += lineHeight - 4;
            }
            yPos += sectionGap;
        }

        // =====================================================================
        // Genome Traits Section
        // =====================================================================
        const genome = agent.genome || {};
        const traitEntries = Object.entries(genome);

        if (traitEntries.length > 0) {
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px sans-serif';
            ctx.fillText('Genome Traits', panelX + padding, yPos);
            yPos += lineHeight + 4;

            const barWidth = PANEL_WIDTH - padding * 2 - 100;
            const barHeight = 12;
            const labelWidth = 95;

            for (const [trait, value] of traitEntries) {
                // Trait name
                ctx.fillStyle = '#cccccc';
                ctx.font = '11px sans-serif';
                let traitName = trait.replace(/_/g, ' ');
                // Truncate long names
                if (traitName.length > 14) {
                    traitName = traitName.substring(0, 13) + '...';
                }
                ctx.fillText(traitName, panelX + padding, yPos + 1);

                // Bar background
                const barX = panelX + padding + labelWidth;
                ctx.fillStyle = 'rgba(60, 60, 80, 0.8)';
                ctx.fillRect(barX, yPos, barWidth, barHeight);

                // Bar fill
                const fillWidth = Math.max(0, Math.min(1, value)) * barWidth;
                ctx.fillStyle = getTraitBarColor(value);
                ctx.fillRect(barX, yPos, fillWidth, barHeight);

                // Value text
                ctx.fillStyle = '#ffffff';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(value.toFixed(2), barX + barWidth - 3, yPos + 1);
                ctx.textAlign = 'left';

                yPos += barHeight + 6;
            }
            yPos += sectionGap;
        }

        // =====================================================================
        // Connected Links Section
        // =====================================================================
        if (currentFrame && currentFrame.links) {
            const connectedLinks = currentFrame.links.filter(link =>
                link.source_id === agent.id || link.dest_id === agent.id
            );

            if (connectedLinks.length > 0) {
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 13px sans-serif';
                ctx.fillText('Connected Links', panelX + padding, yPos);
                yPos += lineHeight + 4;

                for (const link of connectedLinks) {
                    // Determine direction
                    const isSource = link.source_id === agent.id;
                    const otherAgentId = isSource ? link.dest_id : link.source_id;
                    const arrow = isSource ? '→' : '←';

                    // Find the other agent's name
                    let otherName = otherAgentId;
                    if (currentFrame.agents) {
                        const otherAgent = currentFrame.agents.find(a => a.id === otherAgentId);
                        if (otherAgent) {
                            otherName = otherAgent.name;
                        }
                    }

                    // Link type color
                    ctx.fillStyle = getLinkTypeColor(link.link_type);
                    ctx.font = '11px sans-serif';
                    ctx.fillText(arrow + ' ' + otherName, panelX + padding + 8, yPos);

                    // Link type label
                    ctx.fillStyle = '#888888';
                    ctx.font = '10px sans-serif';
                    const typeLabel = '(' + link.link_type + ')';
                    ctx.fillText(typeLabel, panelX + padding + 120, yPos);

                    yPos += lineHeight - 2;
                }
            }
        }

        ctx.restore();
    }

    /**
     * Get color for link type.
     * @param {string} linkType - Link type
     * @returns {string} Color string
     */
    function getLinkTypeColor(linkType) {
        switch (linkType) {
            case 'hierarchical': return '#e74c3c';  // Red
            case 'peer': return '#3498db';          // Blue
            case 'service': return '#2ecc71';       // Green
            case 'competitive': return '#f39c12';   // Orange
            default: return '#aaaaaa';
        }
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineInteraction = {
        init: init,
        setFrame: setFrame,
        renderTooltip: renderTooltip,
        renderSelectionRing: renderSelectionRing,
        renderDetailPanel: renderDetailPanel,
        getAgentHoverState: getAgentHoverState,
        isLinkHighlighted: isLinkHighlighted,
        getHoveredAgent: getHoveredAgent,
        getSelectedAgent: getSelectedAgent,
        selectAgent: selectAgent,
        deselectAgent: deselectAgent,
        hitTestAgents: hitTestAgents,
        updateAnimations: updateAnimations
    };

})(typeof window !== 'undefined' ? window : this);
