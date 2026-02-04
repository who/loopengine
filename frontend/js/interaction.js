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

    // Zoom and pan configuration
    const MIN_ZOOM = 0.1;   // Macro view - full topology visible
    const MAX_ZOOM = 5.0;   // Micro view - details readable
    const ZOOM_SENSITIVITY = 0.001;  // Mouse wheel sensitivity
    const ZOOM_ANIMATION_SPEED = 0.15;  // Smooth zoom animation speed
    const PAN_FRICTION = 0.92;  // Momentum friction for smooth pan

    // Zoom/pan state
    let isDragging = false;
    let wasDragging = false;  // Track if a drag gesture just ended (for click suppression)
    let dragStartX = 0;
    let dragStartY = 0;
    let viewportStartOffsetX = 0;
    let viewportStartOffsetY = 0;
    let zoomAnimation = null;  // {startScale, targetScale, centerX, centerY, progress}
    let panVelocityX = 0;
    let panVelocityY = 0;
    let lastDragX = 0;
    let lastDragY = 0;

    // Touch state for pinch zoom
    let touchStartDist = 0;
    let touchStartScale = 1;
    let touchCenterX = 0;
    let touchCenterY = 0;
    let isTouchZooming = false;

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * Initialize interaction module with canvas element.
     * @param {HTMLCanvasElement} canvasElement - The canvas element
     */
    function init(canvasElement) {
        canvas = canvasElement;

        // Mouse events for hover/click
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseleave', handleMouseLeave);
        canvas.addEventListener('click', handleClick);

        // Mouse events for zoom and pan
        canvas.addEventListener('wheel', handleWheel, { passive: false });
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mouseup', handleMouseUp);

        // Touch events for pinch zoom
        canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
        canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
        canvas.addEventListener('touchend', handleTouchEnd);
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

        // Handle dragging for pan
        if (isDragging) {
            const deltaX = event.clientX - dragStartX;
            const deltaY = event.clientY - dragStartY;

            // Track velocity for momentum
            panVelocityX = event.clientX - lastDragX;
            panVelocityY = event.clientY - lastDragY;
            lastDragX = event.clientX;
            lastDragY = event.clientY;

            if (typeof LoopEngineRenderer !== 'undefined') {
                LoopEngineRenderer.setViewport({
                    offsetX: viewportStartOffsetX + deltaX,
                    offsetY: viewportStartOffsetY + deltaY
                });
            }
            return;  // Don't update hover while dragging
        }

        // Perform hit test against agents
        hoveredAgent = hitTestAgents(mouseX, mouseY);

        // Update cursor based on what's under mouse
        if (hoveredAgent) {
            canvas.style.cursor = 'pointer';
        } else {
            canvas.style.cursor = 'grab';
        }
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
        // Ignore clicks if we just finished a drag gesture
        if (wasDragging) {
            wasDragging = false;
            return;
        }

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

    // =========================================================================
    // Zoom and Pan Event Handlers
    // =========================================================================

    /**
     * Handle mouse wheel for zoom.
     * @param {WheelEvent} event - Wheel event
     */
    function handleWheel(event) {
        event.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        // Calculate zoom factor
        const delta = -event.deltaY * ZOOM_SENSITIVITY;
        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        // Calculate new scale with limits
        const currentScale = viewport.scale;
        const targetScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, currentScale * (1 + delta)));

        // Initiate smooth zoom animation centered on mouse
        startZoomAnimation(currentScale, targetScale, mouseX, mouseY);
    }

    /**
     * Start a smooth zoom animation.
     * @param {number} startScale - Starting scale
     * @param {number} targetScale - Target scale
     * @param {number} centerX - Zoom center X (screen coords)
     * @param {number} centerY - Zoom center Y (screen coords)
     */
    function startZoomAnimation(startScale, targetScale, centerX, centerY) {
        zoomAnimation = {
            startScale: startScale,
            targetScale: targetScale,
            centerX: centerX,
            centerY: centerY,
            progress: 0
        };
    }

    /**
     * Handle mouse down for pan start.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseDown(event) {
        // Only pan with left button on empty space
        if (event.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;

        // Check if clicking on an agent
        const hitAgent = hitTestAgents(clickX, clickY);
        if (hitAgent) {
            // Don't start pan if clicking an agent
            return;
        }

        isDragging = true;
        dragStartX = event.clientX;
        dragStartY = event.clientY;
        lastDragX = event.clientX;
        lastDragY = event.clientY;

        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        viewportStartOffsetX = viewport.offsetX;
        viewportStartOffsetY = viewport.offsetY;

        // Reset velocity
        panVelocityX = 0;
        panVelocityY = 0;

        // Change cursor
        canvas.style.cursor = 'grabbing';
    }

    /**
     * Handle mouse up for pan end.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseUp(event) {
        if (isDragging) {
            isDragging = false;

            // Check if this was a significant drag vs a click
            const dragDist = Math.sqrt(
                Math.pow(event.clientX - dragStartX, 2) +
                Math.pow(event.clientY - dragStartY, 2)
            );

            // If it was a significant drag, suppress the click event
            if (dragDist >= 5) {
                wasDragging = true;
            }

            // Restore cursor based on what's under mouse
            const rect = canvas.getBoundingClientRect();
            const mx = event.clientX - rect.left;
            const my = event.clientY - rect.top;
            const hitAgent = hitTestAgents(mx, my);
            canvas.style.cursor = hitAgent ? 'pointer' : 'grab';
        }
    }

    // =========================================================================
    // Touch Event Handlers for Pinch Zoom
    // =========================================================================

    /**
     * Get distance between two touch points.
     * @param {Touch} touch1 - First touch
     * @param {Touch} touch2 - Second touch
     * @returns {number} Distance in pixels
     */
    function getTouchDistance(touch1, touch2) {
        const dx = touch1.clientX - touch2.clientX;
        const dy = touch1.clientY - touch2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Get center point between two touches.
     * @param {Touch} touch1 - First touch
     * @param {Touch} touch2 - Second touch
     * @param {DOMRect} rect - Canvas bounding rect
     * @returns {Object} {x, y} center in canvas coords
     */
    function getTouchCenter(touch1, touch2, rect) {
        return {
            x: ((touch1.clientX + touch2.clientX) / 2) - rect.left,
            y: ((touch1.clientY + touch2.clientY) / 2) - rect.top
        };
    }

    /**
     * Handle touch start for pinch zoom.
     * @param {TouchEvent} event - Touch event
     */
    function handleTouchStart(event) {
        if (event.touches.length === 2) {
            event.preventDefault();
            isTouchZooming = true;

            touchStartDist = getTouchDistance(event.touches[0], event.touches[1]);

            const viewport = typeof LoopEngineRenderer !== 'undefined'
                ? LoopEngineRenderer.getViewport()
                : { scale: 1, offsetX: 0, offsetY: 0 };

            touchStartScale = viewport.scale;

            const rect = canvas.getBoundingClientRect();
            const center = getTouchCenter(event.touches[0], event.touches[1], rect);
            touchCenterX = center.x;
            touchCenterY = center.y;
        } else if (event.touches.length === 1) {
            // Single touch - start pan
            const rect = canvas.getBoundingClientRect();
            const touchX = event.touches[0].clientX - rect.left;
            const touchY = event.touches[0].clientY - rect.top;

            const hitAgent = hitTestAgents(touchX, touchY);
            if (!hitAgent) {
                isDragging = true;
                dragStartX = event.touches[0].clientX;
                dragStartY = event.touches[0].clientY;
                lastDragX = event.touches[0].clientX;
                lastDragY = event.touches[0].clientY;

                const viewport = typeof LoopEngineRenderer !== 'undefined'
                    ? LoopEngineRenderer.getViewport()
                    : { scale: 1, offsetX: 0, offsetY: 0 };

                viewportStartOffsetX = viewport.offsetX;
                viewportStartOffsetY = viewport.offsetY;
                panVelocityX = 0;
                panVelocityY = 0;
            }
        }
    }

    /**
     * Handle touch move for pinch zoom and pan.
     * @param {TouchEvent} event - Touch event
     */
    function handleTouchMove(event) {
        if (event.touches.length === 2 && isTouchZooming) {
            event.preventDefault();

            const currentDist = getTouchDistance(event.touches[0], event.touches[1]);
            const scaleRatio = currentDist / touchStartDist;
            const newScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, touchStartScale * scaleRatio));

            // Apply zoom centered on pinch center
            applyZoomAtPoint(newScale, touchCenterX, touchCenterY);

        } else if (event.touches.length === 1 && isDragging && !isTouchZooming) {
            event.preventDefault();

            const touch = event.touches[0];
            const deltaX = touch.clientX - dragStartX;
            const deltaY = touch.clientY - dragStartY;

            // Track velocity for momentum
            panVelocityX = touch.clientX - lastDragX;
            panVelocityY = touch.clientY - lastDragY;
            lastDragX = touch.clientX;
            lastDragY = touch.clientY;

            if (typeof LoopEngineRenderer !== 'undefined') {
                LoopEngineRenderer.setViewport({
                    offsetX: viewportStartOffsetX + deltaX,
                    offsetY: viewportStartOffsetY + deltaY
                });
            }
        }
    }

    /**
     * Handle touch end.
     * @param {TouchEvent} event - Touch event
     */
    function handleTouchEnd(event) {
        if (event.touches.length < 2) {
            isTouchZooming = false;
        }
        if (event.touches.length === 0) {
            isDragging = false;
        }
    }

    /**
     * Apply zoom at a specific point (for immediate zoom without animation).
     * @param {number} newScale - New scale value
     * @param {number} centerX - Zoom center X (screen coords)
     * @param {number} centerY - Zoom center Y (screen coords)
     */
    function applyZoomAtPoint(newScale, centerX, centerY) {
        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1, offsetX: 0, offsetY: 0 };

        const oldScale = viewport.scale;
        if (newScale === oldScale) return;

        // Calculate world position under cursor before zoom
        const worldX = (centerX - viewport.offsetX) / oldScale;
        const worldY = (centerY - viewport.offsetY) / oldScale;

        // Calculate new offset to keep world position under cursor
        const newOffsetX = centerX - worldX * newScale;
        const newOffsetY = centerY - worldY * newScale;

        if (typeof LoopEngineRenderer !== 'undefined') {
            LoopEngineRenderer.setViewport({
                scale: newScale,
                offsetX: newOffsetX,
                offsetY: newOffsetY
            });
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

        // Update pan animation (for centering on selected agent)
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

        // Update zoom animation
        if (zoomAnimation && zoomAnimation.progress < 1) {
            zoomAnimation.progress = Math.min(1, zoomAnimation.progress + ZOOM_ANIMATION_SPEED);

            const t = easeOutCubic(zoomAnimation.progress);
            const newScale = lerp(zoomAnimation.startScale, zoomAnimation.targetScale, t);

            // Apply zoom centered on the animation center point
            applyZoomAtPoint(newScale, zoomAnimation.centerX, zoomAnimation.centerY);

            if (zoomAnimation.progress >= 1) {
                zoomAnimation = null;
            }
        }

        // Apply pan momentum (when not dragging)
        if (!isDragging && (Math.abs(panVelocityX) > 0.1 || Math.abs(panVelocityY) > 0.1)) {
            const viewport = typeof LoopEngineRenderer !== 'undefined'
                ? LoopEngineRenderer.getViewport()
                : { scale: 1, offsetX: 0, offsetY: 0 };

            if (typeof LoopEngineRenderer !== 'undefined') {
                LoopEngineRenderer.setViewport({
                    offsetX: viewport.offsetX + panVelocityX,
                    offsetY: viewport.offsetY + panVelocityY
                });
            }

            // Apply friction
            panVelocityX *= PAN_FRICTION;
            panVelocityY *= PAN_FRICTION;

            // Stop if velocity is very small
            if (Math.abs(panVelocityX) < 0.1) panVelocityX = 0;
            if (Math.abs(panVelocityY) < 0.1) panVelocityY = 0;
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

    /**
     * Zoom to fit all content in view (macro view).
     */
    function zoomToFit() {
        if (!currentFrame || !currentFrame.agents || currentFrame.agents.length === 0) return;

        const canvasWidth = canvas._logicalWidth || canvas.width;
        const canvasHeight = canvas._logicalHeight || canvas.height;

        if (typeof LoopEngineRenderer !== 'undefined') {
            LoopEngineRenderer.autoCenterViewport(currentFrame.agents, canvasWidth, canvasHeight);
        }
    }

    /**
     * Get current zoom level.
     * @returns {number} Current scale
     */
    function getZoomLevel() {
        const viewport = typeof LoopEngineRenderer !== 'undefined'
            ? LoopEngineRenderer.getViewport()
            : { scale: 1 };
        return viewport.scale;
    }

    /**
     * Set zoom level programmatically.
     * @param {number} scale - New scale (clamped to MIN_ZOOM/MAX_ZOOM)
     * @param {boolean} animate - Whether to animate the transition
     */
    function setZoomLevel(scale, animate) {
        const clampedScale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, scale));
        const canvasWidth = canvas._logicalWidth || canvas.width;
        const canvasHeight = canvas._logicalHeight || canvas.height;

        if (animate) {
            const viewport = typeof LoopEngineRenderer !== 'undefined'
                ? LoopEngineRenderer.getViewport()
                : { scale: 1 };
            startZoomAnimation(viewport.scale, clampedScale, canvasWidth / 2, canvasHeight / 2);
        } else {
            applyZoomAtPoint(clampedScale, canvasWidth / 2, canvasHeight / 2);
        }
    }

    /**
     * Check if currently dragging (panning).
     * @returns {boolean} True if dragging
     */
    function isDraggingViewport() {
        return isDragging;
    }

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
        updateAnimations: updateAnimations,
        // Zoom and pan controls
        zoomToFit: zoomToFit,
        getZoomLevel: getZoomLevel,
        setZoomLevel: setZoomLevel,
        isDraggingViewport: isDraggingViewport,
        // Zoom constants for external reference
        MIN_ZOOM: MIN_ZOOM,
        MAX_ZOOM: MAX_ZOOM
    };

})(typeof window !== 'undefined' ? window : this);
