/**
 * LoopEngine GA (Genetic Algorithm) Progress Module
 *
 * Provides GA progress visualization with generation counter, fitness graph,
 * role selector, and start/stop controls.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================

    const PANEL_WIDTH = 280;
    const PANEL_PADDING = 16;
    const GRAPH_HEIGHT = 120;
    const MAX_GRAPH_POINTS = 100;

    // Colors
    const PANEL_BG_COLOR = 'rgba(26, 26, 46, 0.95)';
    const PANEL_BORDER_COLOR = 'rgba(100, 100, 120, 0.5)';
    const TEXT_COLOR = '#cccccc';
    const LABEL_COLOR = '#888888';
    const BUTTON_COLOR = '#4a90d9';
    const BUTTON_HOVER_COLOR = '#5da0e9';
    const BUTTON_DISABLED_COLOR = '#555555';
    const STOP_BUTTON_COLOR = '#d94a4a';
    const STOP_BUTTON_HOVER_COLOR = '#e95d5d';
    const GRAPH_LINE_COLOR = '#4a90d9';
    const GRAPH_FILL_COLOR = 'rgba(74, 144, 217, 0.2)';
    const GRAPH_GRID_COLOR = 'rgba(100, 100, 120, 0.3)';
    const INPUT_BG_COLOR = 'rgba(40, 40, 60, 0.8)';
    const INPUT_BORDER_COLOR = 'rgba(100, 100, 120, 0.5)';
    const SELECT_BG_COLOR = 'rgba(40, 40, 60, 0.8)';

    // Available roles for evolution
    const AVAILABLE_ROLES = [
        { value: 'sandwich_maker', label: 'Sandwich Maker (Tom)' },
        { value: 'cashier', label: 'Cashier (Alex)' },
        { value: 'owner', label: 'Owner (Maria)' }
    ];

    // =========================================================================
    // State
    // =========================================================================

    let canvas = null;
    let sendCommand = null;
    let panelVisible = true;

    // GA state
    let isRunning = false;
    let currentJobId = null;
    let currentGeneration = 0;
    let totalGenerations = 100;
    let bestFitness = null;
    let fitnessHistory = [];
    let selectedRole = AVAILABLE_ROLES[0].value;
    let generationInput = 100;
    let bestGenome = null;

    // UI interaction state
    let isHoveringStart = false;
    let isHoveringStop = false;
    let isHoveringToggle = false;

    // UI element positions (calculated during render)
    let panelRect = { x: 0, y: 0, width: PANEL_WIDTH, height: 0 };
    let startButtonRect = { x: 0, y: 0, width: 0, height: 0 };
    let stopButtonRect = { x: 0, y: 0, width: 0, height: 0 };
    let toggleButtonRect = { x: 0, y: 0, width: 24, height: 24 };
    let roleSelectRect = { x: 0, y: 0, width: 0, height: 0 };
    let generationInputRect = { x: 0, y: 0, width: 0, height: 0 };

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * Initialize the GA module.
     * @param {HTMLCanvasElement} canvasElement - The canvas element
     * @param {Function} commandCallback - Callback to send control commands
     */
    function init(canvasElement, commandCallback) {
        canvas = canvasElement;
        sendCommand = commandCallback;

        // Add event listeners
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);

        // Request initial GA status
        if (sendCommand) {
            sendCommand('get_ga_status');
        }
    }

    /**
     * Cleanup event listeners.
     */
    function destroy() {
        if (canvas) {
            canvas.removeEventListener('mousedown', handleMouseDown);
            canvas.removeEventListener('mousemove', handleMouseMove);
            canvas.removeEventListener('mouseup', handleMouseUp);
        }
        canvas = null;
        sendCommand = null;
    }

    // =========================================================================
    // Event Handlers
    // =========================================================================

    /**
     * Get mouse position relative to canvas.
     * @param {MouseEvent} event - Mouse event
     * @returns {Object} {x, y} position
     */
    function getMousePos(event) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }

    /**
     * Check if point is inside a rectangle.
     * @param {number} px - Point X
     * @param {number} py - Point Y
     * @param {Object} rect - Rectangle {x, y, width, height}
     * @returns {boolean} True if point is inside
     */
    function isInsideRect(px, py, rect) {
        return px >= rect.x && px <= rect.x + rect.width &&
               py >= rect.y && py <= rect.y + rect.height;
    }

    /**
     * Handle mouse down event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseDown(event) {
        const pos = getMousePos(event);

        // Check toggle button
        if (isInsideRect(pos.x, pos.y, toggleButtonRect)) {
            panelVisible = !panelVisible;
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        if (!panelVisible) return;
        if (!isInsideRect(pos.x, pos.y, panelRect)) return;

        // Check start button
        if (!isRunning && isInsideRect(pos.x, pos.y, startButtonRect)) {
            startGA();
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // Check stop button
        if (isRunning && isInsideRect(pos.x, pos.y, stopButtonRect)) {
            stopGA();
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // Check role selector (open prompt dialog)
        if (!isRunning && isInsideRect(pos.x, pos.y, roleSelectRect)) {
            cycleRole();
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // Check generation input (open prompt dialog)
        if (!isRunning && isInsideRect(pos.x, pos.y, generationInputRect)) {
            promptGenerations();
            event.preventDefault();
            event.stopPropagation();
            return;
        }
    }

    /**
     * Handle mouse move event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseMove(event) {
        const pos = getMousePos(event);

        // Update hover states
        isHoveringToggle = isInsideRect(pos.x, pos.y, toggleButtonRect);

        if (!panelVisible) {
            isHoveringStart = false;
            isHoveringStop = false;
            return;
        }

        if (!isInsideRect(pos.x, pos.y, panelRect)) {
            isHoveringStart = false;
            isHoveringStop = false;
            return;
        }

        isHoveringStart = !isRunning && isInsideRect(pos.x, pos.y, startButtonRect);
        isHoveringStop = isRunning && isInsideRect(pos.x, pos.y, stopButtonRect);

        // Update cursor
        if (isHoveringStart || isHoveringStop || isHoveringToggle ||
            (!isRunning && (isInsideRect(pos.x, pos.y, roleSelectRect) ||
                           isInsideRect(pos.x, pos.y, generationInputRect)))) {
            canvas.style.cursor = 'pointer';
        }
    }

    /**
     * Handle mouse up event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseUp(event) {
        // No dragging behavior in GA panel
    }

    /**
     * Cycle through available roles.
     */
    function cycleRole() {
        const currentIndex = AVAILABLE_ROLES.findIndex(r => r.value === selectedRole);
        const nextIndex = (currentIndex + 1) % AVAILABLE_ROLES.length;
        selectedRole = AVAILABLE_ROLES[nextIndex].value;
    }

    /**
     * Prompt user for generation count.
     */
    function promptGenerations() {
        const input = prompt('Enter number of generations (1-1000):', generationInput);
        if (input !== null) {
            const value = parseInt(input, 10);
            if (!isNaN(value) && value >= 1 && value <= 1000) {
                generationInput = value;
                totalGenerations = value;
            }
        }
    }

    // =========================================================================
    // GA Control Actions
    // =========================================================================

    /**
     * Start GA evolution.
     */
    function startGA() {
        if (isRunning || !sendCommand) return;

        // Reset state
        fitnessHistory = [];
        currentGeneration = 0;
        bestFitness = null;
        bestGenome = null;
        totalGenerations = generationInput;

        sendCommand('start_ga', {
            role: selectedRole,
            generations: generationInput,
            population_size: 50
        });
    }

    /**
     * Stop GA evolution.
     */
    function stopGA() {
        if (!isRunning || !sendCommand) return;

        sendCommand('stop_ga', currentJobId ? { job_id: currentJobId } : {});
    }

    // =========================================================================
    // WebSocket Message Handlers
    // =========================================================================

    /**
     * Handle GA progress message from server.
     * @param {Object} data - Progress data {job_id, generation, best_fitness}
     */
    function handleGAProgress(data) {
        currentJobId = data.job_id;
        currentGeneration = data.generation;
        bestFitness = data.best_fitness;
        isRunning = true;

        // Add to fitness history (limit to MAX_GRAPH_POINTS)
        fitnessHistory.push({
            generation: data.generation,
            fitness: data.best_fitness
        });

        if (fitnessHistory.length > MAX_GRAPH_POINTS) {
            // Downsample history
            const step = Math.ceil(fitnessHistory.length / MAX_GRAPH_POINTS);
            const downsampled = [];
            for (let i = 0; i < fitnessHistory.length; i += step) {
                downsampled.push(fitnessHistory[i]);
            }
            fitnessHistory = downsampled;
        }
    }

    /**
     * Handle GA complete message from server.
     * @param {Object} data - Completion data {job_id, best_genome}
     */
    function handleGAComplete(data) {
        isRunning = false;
        bestGenome = data.best_genome;
        console.log('GA completed. Best genome:', bestGenome);
    }

    /**
     * Handle GA status response.
     * @param {Object} data - Status response from get_ga_status
     */
    function handleGAStatus(data) {
        if (!data.success) return;

        if (data.status === 'running') {
            isRunning = true;
            currentJobId = data.job_id;
            currentGeneration = data.current_generation || 0;
            totalGenerations = data.total_generations || 100;
            bestFitness = data.best_fitness;
            bestGenome = data.best_genome;
        } else if (data.status === 'completed') {
            isRunning = false;
            bestGenome = data.best_genome;
            bestFitness = data.best_fitness;
        } else {
            isRunning = false;
        }
    }

    /**
     * Handle start_ga response (success or failure).
     * @param {Object} data - Response from start_ga command
     */
    function handleStartGAResponse(data) {
        if (data.success) {
            isRunning = true;
            currentJobId = data.job_id;
        } else {
            console.error('Failed to start GA:', data.message);
        }
    }

    /**
     * Handle stop_ga response.
     * @param {Object} data - Response from stop_ga command
     */
    function handleStopGAResponse(data) {
        if (data.success) {
            // GA will send ga_complete when it actually stops
            console.log('GA stop requested');
        }
    }

    // =========================================================================
    // Rendering
    // =========================================================================

    /**
     * Render the GA panel.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     */
    function renderGAPanel(ctx, canvasWidth, canvasHeight) {
        // Render toggle button (always visible)
        renderToggleButton(ctx, canvasWidth);

        if (!panelVisible) return;

        // Calculate panel position (top-right)
        const panelX = canvasWidth - PANEL_WIDTH - 12;
        const panelY = 12;

        // Calculate panel height based on content
        let contentHeight = PANEL_PADDING; // Top padding

        // Title
        contentHeight += 24;
        contentHeight += 12;

        // Role selector
        contentHeight += 16; // Label
        contentHeight += 32; // Select box
        contentHeight += 12;

        // Generations input
        contentHeight += 16; // Label
        contentHeight += 32; // Input box
        contentHeight += 16;

        // Buttons
        contentHeight += 36;
        contentHeight += 16;

        // Status (when running or completed)
        if (isRunning || fitnessHistory.length > 0) {
            contentHeight += 16; // Generation label
            contentHeight += 24; // Generation value
            contentHeight += 8;
            contentHeight += 16; // Best fitness label
            contentHeight += 24; // Best fitness value
            contentHeight += 16;
        }

        // Graph (when we have history)
        if (fitnessHistory.length > 0) {
            contentHeight += GRAPH_HEIGHT;
            contentHeight += 12;
        }

        contentHeight += PANEL_PADDING; // Bottom padding

        panelRect = { x: panelX, y: panelY, width: PANEL_WIDTH, height: contentHeight };

        // Draw panel background
        ctx.fillStyle = PANEL_BG_COLOR;
        ctx.beginPath();
        ctx.roundRect(panelRect.x, panelRect.y, panelRect.width, panelRect.height, 8);
        ctx.fill();

        // Draw panel border
        ctx.strokeStyle = PANEL_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw content
        let y = panelY + PANEL_PADDING;
        const contentX = panelX + PANEL_PADDING;
        const contentWidth = PANEL_WIDTH - PANEL_PADDING * 2;

        // Title
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = 'bold 14px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('Genetic Algorithm', contentX, y);
        y += 24 + 12;

        // Role selector
        ctx.fillStyle = LABEL_COLOR;
        ctx.font = '11px monospace';
        ctx.fillText('Role to Evolve', contentX, y);
        y += 16;

        roleSelectRect = { x: contentX, y: y, width: contentWidth, height: 28 };
        renderSelectBox(ctx, roleSelectRect, selectedRole, AVAILABLE_ROLES, !isRunning);
        y += 32 + 12;

        // Generations input
        ctx.fillStyle = LABEL_COLOR;
        ctx.font = '11px monospace';
        ctx.fillText('Generations', contentX, y);
        y += 16;

        generationInputRect = { x: contentX, y: y, width: contentWidth, height: 28 };
        renderInputBox(ctx, generationInputRect, generationInput.toString(), !isRunning);
        y += 32 + 16;

        // Buttons
        const buttonWidth = (contentWidth - 8) / 2;

        startButtonRect = { x: contentX, y: y, width: buttonWidth, height: 32 };
        stopButtonRect = { x: contentX + buttonWidth + 8, y: y, width: buttonWidth, height: 32 };

        renderButton(ctx, startButtonRect, 'Start', !isRunning, isHoveringStart, BUTTON_COLOR, BUTTON_HOVER_COLOR);
        renderButton(ctx, stopButtonRect, 'Stop', isRunning, isHoveringStop, STOP_BUTTON_COLOR, STOP_BUTTON_HOVER_COLOR);
        y += 36 + 16;

        // Status (when running or completed)
        if (isRunning || fitnessHistory.length > 0) {
            // Generation counter
            ctx.fillStyle = LABEL_COLOR;
            ctx.font = '11px monospace';
            ctx.fillText('Generation', contentX, y);
            y += 16;

            ctx.fillStyle = TEXT_COLOR;
            ctx.font = 'bold 18px monospace';
            const genText = isRunning ? `${currentGeneration} / ${totalGenerations}` : `${currentGeneration}`;
            ctx.fillText(genText, contentX, y);
            y += 24 + 8;

            // Best fitness
            ctx.fillStyle = LABEL_COLOR;
            ctx.font = '11px monospace';
            ctx.fillText('Best Fitness', contentX, y);
            y += 16;

            ctx.fillStyle = TEXT_COLOR;
            ctx.font = 'bold 18px monospace';
            const fitnessText = bestFitness !== null ? bestFitness.toFixed(4) : '--';
            ctx.fillText(fitnessText, contentX, y);
            y += 24 + 16;
        }

        // Fitness graph
        if (fitnessHistory.length > 0) {
            const graphRect = { x: contentX, y: y, width: contentWidth, height: GRAPH_HEIGHT };
            renderFitnessGraph(ctx, graphRect);
        }
    }

    /**
     * Render toggle button to show/hide panel.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} canvasWidth - Canvas width
     */
    function renderToggleButton(ctx, canvasWidth) {
        toggleButtonRect = {
            x: canvasWidth - 36,
            y: panelVisible ? panelRect.y + panelRect.height + 8 : 12,
            width: 24,
            height: 24
        };

        ctx.fillStyle = isHoveringToggle ? BUTTON_HOVER_COLOR : BUTTON_COLOR;
        ctx.beginPath();
        ctx.roundRect(toggleButtonRect.x, toggleButtonRect.y, toggleButtonRect.width, toggleButtonRect.height, 4);
        ctx.fill();

        // Draw GA icon
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('GA', toggleButtonRect.x + 12, toggleButtonRect.y + 12);
    }

    /**
     * Render a select box.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Rectangle
     * @param {string} value - Current value
     * @param {Array} options - Available options
     * @param {boolean} enabled - Whether select is enabled
     */
    function renderSelectBox(ctx, rect, value, options, enabled) {
        ctx.fillStyle = enabled ? SELECT_BG_COLOR : 'rgba(30, 30, 45, 0.6)';
        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 4);
        ctx.fill();

        ctx.strokeStyle = INPUT_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Find label for value
        const option = options.find(o => o.value === value);
        const label = option ? option.label : value;

        ctx.fillStyle = enabled ? TEXT_COLOR : LABEL_COLOR;
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, rect.x + 8, rect.y + rect.height / 2);

        // Draw dropdown arrow
        const arrowX = rect.x + rect.width - 16;
        const arrowY = rect.y + rect.height / 2;
        ctx.fillStyle = enabled ? TEXT_COLOR : LABEL_COLOR;
        ctx.beginPath();
        ctx.moveTo(arrowX - 4, arrowY - 2);
        ctx.lineTo(arrowX + 4, arrowY - 2);
        ctx.lineTo(arrowX, arrowY + 4);
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Render an input box.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Rectangle
     * @param {string} value - Current value
     * @param {boolean} enabled - Whether input is enabled
     */
    function renderInputBox(ctx, rect, value, enabled) {
        ctx.fillStyle = enabled ? INPUT_BG_COLOR : 'rgba(30, 30, 45, 0.6)';
        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 4);
        ctx.fill();

        ctx.strokeStyle = INPUT_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = enabled ? TEXT_COLOR : LABEL_COLOR;
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(value, rect.x + 8, rect.y + rect.height / 2);
    }

    /**
     * Render a button.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Rectangle
     * @param {string} label - Button label
     * @param {boolean} enabled - Whether button is enabled
     * @param {boolean} hovering - Whether mouse is hovering
     * @param {string} color - Normal color
     * @param {string} hoverColor - Hover color
     */
    function renderButton(ctx, rect, label, enabled, hovering, color, hoverColor) {
        if (enabled) {
            ctx.fillStyle = hovering ? hoverColor : color;
        } else {
            ctx.fillStyle = BUTTON_DISABLED_COLOR;
        }

        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 4);
        ctx.fill();

        ctx.fillStyle = enabled ? '#ffffff' : '#888888';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    /**
     * Render fitness graph.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Graph rectangle
     */
    function renderFitnessGraph(ctx, rect) {
        if (fitnessHistory.length === 0) return;

        // Draw background
        ctx.fillStyle = 'rgba(20, 20, 35, 0.5)';
        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 4);
        ctx.fill();

        const padding = 8;
        const graphX = rect.x + padding;
        const graphY = rect.y + padding;
        const graphWidth = rect.width - padding * 2;
        const graphHeight = rect.height - padding * 2;

        // Calculate min/max fitness
        let minFitness = Infinity;
        let maxFitness = -Infinity;
        for (const point of fitnessHistory) {
            if (point.fitness < minFitness) minFitness = point.fitness;
            if (point.fitness > maxFitness) maxFitness = point.fitness;
        }

        // Add some padding to range
        const range = maxFitness - minFitness;
        if (range > 0) {
            minFitness -= range * 0.1;
            maxFitness += range * 0.1;
        } else {
            minFitness -= 0.1;
            maxFitness += 0.1;
        }

        // Draw horizontal grid lines
        ctx.strokeStyle = GRAPH_GRID_COLOR;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        for (let i = 0; i <= 4; i++) {
            const y = graphY + (i / 4) * graphHeight;
            ctx.beginPath();
            ctx.moveTo(graphX, y);
            ctx.lineTo(graphX + graphWidth, y);
            ctx.stroke();
        }
        ctx.setLineDash([]);

        // Draw fitness line
        if (fitnessHistory.length > 1) {
            // Draw fill
            ctx.fillStyle = GRAPH_FILL_COLOR;
            ctx.beginPath();
            ctx.moveTo(graphX, graphY + graphHeight);

            for (let i = 0; i < fitnessHistory.length; i++) {
                const x = graphX + (i / (fitnessHistory.length - 1)) * graphWidth;
                const normalizedY = (fitnessHistory[i].fitness - minFitness) / (maxFitness - minFitness);
                const y = graphY + graphHeight - normalizedY * graphHeight;
                ctx.lineTo(x, y);
            }

            ctx.lineTo(graphX + graphWidth, graphY + graphHeight);
            ctx.closePath();
            ctx.fill();

            // Draw line
            ctx.strokeStyle = GRAPH_LINE_COLOR;
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < fitnessHistory.length; i++) {
                const x = graphX + (i / (fitnessHistory.length - 1)) * graphWidth;
                const normalizedY = (fitnessHistory[i].fitness - minFitness) / (maxFitness - minFitness);
                const y = graphY + graphHeight - normalizedY * graphHeight;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw current point
            const lastPoint = fitnessHistory[fitnessHistory.length - 1];
            const lastX = graphX + graphWidth;
            const lastNormalizedY = (lastPoint.fitness - minFitness) / (maxFitness - minFitness);
            const lastY = graphY + graphHeight - lastNormalizedY * graphHeight;

            ctx.fillStyle = GRAPH_LINE_COLOR;
            ctx.beginPath();
            ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw axis labels
        ctx.fillStyle = LABEL_COLOR;
        ctx.font = '9px monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(maxFitness.toFixed(2), graphX - 4, graphY);
        ctx.fillText(minFitness.toFixed(2), graphX - 4, graphY + graphHeight);

        // Draw title
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('Fitness over Generations', rect.x + 4, rect.y + 4);
    }

    /**
     * Check if a point is in the GA panel area.
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {boolean} True if in panel area
     */
    function isPointInPanel(x, y) {
        if (isInsideRect(x, y, toggleButtonRect)) return true;
        if (!panelVisible) return false;
        return isInsideRect(x, y, panelRect);
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineGA = {
        // Initialization
        init: init,
        destroy: destroy,

        // WebSocket message handlers
        handleGAProgress: handleGAProgress,
        handleGAComplete: handleGAComplete,
        handleGAStatus: handleGAStatus,
        handleStartGAResponse: handleStartGAResponse,
        handleStopGAResponse: handleStopGAResponse,

        // Getters
        isRunning: function() { return isRunning; },
        getCurrentGeneration: function() { return currentGeneration; },
        getTotalGenerations: function() { return totalGenerations; },
        getBestFitness: function() { return bestFitness; },
        getBestGenome: function() { return bestGenome; },
        isPointInPanel: isPointInPanel,

        // Rendering
        renderGAPanel: renderGAPanel
    };

})(typeof window !== 'undefined' ? window : this);
