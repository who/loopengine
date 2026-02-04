/**
 * LoopEngine Controls Module
 *
 * Provides play/pause, speed control, and tick counter UI.
 * Control bar rendered at bottom of canvas per PRD section 7.6.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================

    const CONTROL_BAR_HEIGHT = 48;
    const CONTROL_BAR_PADDING = 12;
    const BUTTON_SIZE = 32;
    const SLIDER_WIDTH = 150;
    const SLIDER_HEIGHT = 6;
    const SLIDER_KNOB_SIZE = 14;

    // Speed range
    const MIN_SPEED = 0.25;
    const MAX_SPEED = 10.0;
    const DEFAULT_SPEED = 1.0;

    // Colors
    const BAR_BG_COLOR = 'rgba(26, 26, 46, 0.85)';
    const BAR_BORDER_COLOR = 'rgba(100, 100, 120, 0.3)';
    const BUTTON_COLOR = '#4a90d9';
    const BUTTON_HOVER_COLOR = '#5da0e9';
    const SLIDER_TRACK_COLOR = 'rgba(100, 100, 120, 0.4)';
    const SLIDER_FILL_COLOR = '#4a90d9';
    const SLIDER_KNOB_COLOR = '#ffffff';
    const TEXT_COLOR = '#aaaaaa';

    // =========================================================================
    // State
    // =========================================================================

    let isPlaying = true;
    let currentSpeed = DEFAULT_SPEED;
    let currentTick = 0;
    let canvas = null;
    let sendCommand = null;  // Callback to send WebSocket commands

    // UI interaction state
    let isDraggingSlider = false;
    let isHoveringPlayPause = false;
    let isHoveringSlider = false;

    // UI element positions (calculated in renderControlBar)
    let playPauseButtonRect = { x: 0, y: 0, width: BUTTON_SIZE, height: BUTTON_SIZE };
    let sliderRect = { x: 0, y: 0, width: SLIDER_WIDTH, height: SLIDER_HEIGHT };
    let sliderKnobX = 0;

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * Initialize the controls module.
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
        canvas.addEventListener('mouseleave', handleMouseLeave);
    }

    /**
     * Cleanup event listeners.
     */
    function destroy() {
        if (canvas) {
            canvas.removeEventListener('mousedown', handleMouseDown);
            canvas.removeEventListener('mousemove', handleMouseMove);
            canvas.removeEventListener('mouseup', handleMouseUp);
            canvas.removeEventListener('mouseleave', handleMouseLeave);
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
     * Check if point is in the control bar area.
     * @param {number} y - Y position
     * @param {number} canvasHeight - Canvas height
     * @returns {boolean} True if in control bar
     */
    function isInControlBar(y, canvasHeight) {
        return y >= canvasHeight - CONTROL_BAR_HEIGHT;
    }

    /**
     * Handle mouse down event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseDown(event) {
        const pos = getMousePos(event);
        const canvasHeight = canvas._logicalHeight || canvas.height;

        if (!isInControlBar(pos.y, canvasHeight)) return;

        // Check play/pause button
        if (isInsideRect(pos.x, pos.y, playPauseButtonRect)) {
            togglePlayPause();
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // Check slider area (wider hit area for knob)
        const sliderHitRect = {
            x: sliderRect.x - SLIDER_KNOB_SIZE / 2,
            y: sliderRect.y - SLIDER_KNOB_SIZE,
            width: sliderRect.width + SLIDER_KNOB_SIZE,
            height: sliderRect.height + SLIDER_KNOB_SIZE * 2
        };

        if (isInsideRect(pos.x, pos.y, sliderHitRect)) {
            isDraggingSlider = true;
            updateSpeedFromMouseX(pos.x);
            event.preventDefault();
            event.stopPropagation();
        }
    }

    /**
     * Handle mouse move event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseMove(event) {
        const pos = getMousePos(event);
        const canvasHeight = canvas._logicalHeight || canvas.height;

        if (isDraggingSlider) {
            updateSpeedFromMouseX(pos.x);
            event.preventDefault();
            return;
        }

        if (!isInControlBar(pos.y, canvasHeight)) {
            isHoveringPlayPause = false;
            isHoveringSlider = false;
            return;
        }

        // Update hover states
        isHoveringPlayPause = isInsideRect(pos.x, pos.y, playPauseButtonRect);

        const sliderHitRect = {
            x: sliderRect.x - SLIDER_KNOB_SIZE / 2,
            y: sliderRect.y - SLIDER_KNOB_SIZE,
            width: sliderRect.width + SLIDER_KNOB_SIZE,
            height: sliderRect.height + SLIDER_KNOB_SIZE * 2
        };
        isHoveringSlider = isInsideRect(pos.x, pos.y, sliderHitRect);

        // Update cursor
        if (isHoveringPlayPause || isHoveringSlider) {
            canvas.style.cursor = 'pointer';
        }
    }

    /**
     * Handle mouse up event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseUp(event) {
        if (isDraggingSlider) {
            isDraggingSlider = false;
            event.preventDefault();
        }
    }

    /**
     * Handle mouse leave event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseLeave(event) {
        isDraggingSlider = false;
        isHoveringPlayPause = false;
        isHoveringSlider = false;
    }

    /**
     * Update speed based on mouse X position.
     * @param {number} mouseX - Mouse X coordinate
     */
    function updateSpeedFromMouseX(mouseX) {
        // Calculate normalized position (0-1) within slider
        const normalized = Math.max(0, Math.min(1,
            (mouseX - sliderRect.x) / sliderRect.width
        ));

        // Map to speed range (logarithmic for better feel)
        // Use log scale: MIN_SPEED at 0, 1.0 at ~0.3, MAX_SPEED at 1.0
        const logMin = Math.log(MIN_SPEED);
        const logMax = Math.log(MAX_SPEED);
        const logSpeed = logMin + normalized * (logMax - logMin);
        const newSpeed = Math.exp(logSpeed);

        setSpeed(newSpeed);
    }

    // =========================================================================
    // Control Actions
    // =========================================================================

    /**
     * Toggle play/pause state.
     */
    function togglePlayPause() {
        isPlaying = !isPlaying;
        if (sendCommand) {
            sendCommand(isPlaying ? 'play' : 'pause');
        }
    }

    /**
     * Set play state.
     * @param {boolean} playing - Whether to play
     */
    function setPlaying(playing) {
        isPlaying = playing;
        if (sendCommand) {
            sendCommand(playing ? 'play' : 'pause');
        }
    }

    /**
     * Set simulation speed.
     * @param {number} speed - Speed multiplier (0.25 to 10.0)
     */
    function setSpeed(speed) {
        currentSpeed = Math.max(MIN_SPEED, Math.min(MAX_SPEED, speed));
        if (sendCommand) {
            sendCommand('set_speed', { speed: currentSpeed });
        }
    }

    /**
     * Update the current tick from frame data.
     * @param {number} tick - Current tick number
     */
    function updateTick(tick) {
        currentTick = tick;
    }

    // =========================================================================
    // Rendering
    // =========================================================================

    /**
     * Render the control bar.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     */
    function renderControlBar(ctx, canvasWidth, canvasHeight) {
        const barY = canvasHeight - CONTROL_BAR_HEIGHT;

        // Background
        ctx.fillStyle = BAR_BG_COLOR;
        ctx.fillRect(0, barY, canvasWidth, CONTROL_BAR_HEIGHT);

        // Top border
        ctx.strokeStyle = BAR_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, barY);
        ctx.lineTo(canvasWidth, barY);
        ctx.stroke();

        // Calculate element positions
        const centerY = barY + CONTROL_BAR_HEIGHT / 2;
        let currentX = CONTROL_BAR_PADDING;

        // Play/Pause button
        playPauseButtonRect = {
            x: currentX,
            y: centerY - BUTTON_SIZE / 2,
            width: BUTTON_SIZE,
            height: BUTTON_SIZE
        };
        renderPlayPauseButton(ctx, playPauseButtonRect, isPlaying, isHoveringPlayPause);
        currentX += BUTTON_SIZE + CONTROL_BAR_PADDING;

        // Speed slider
        sliderRect = {
            x: currentX,
            y: centerY - SLIDER_HEIGHT / 2,
            width: SLIDER_WIDTH,
            height: SLIDER_HEIGHT
        };
        renderSpeedSlider(ctx, sliderRect, currentSpeed, centerY);
        currentX += SLIDER_WIDTH + CONTROL_BAR_PADDING;

        // Speed label
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const speedText = currentSpeed.toFixed(2) + 'x';
        ctx.fillText(speedText, currentX, centerY);
        currentX += ctx.measureText(speedText).width + CONTROL_BAR_PADDING * 2;

        // Tick counter (right-aligned)
        const tickText = 'Tick: ' + currentTick;
        ctx.textAlign = 'right';
        ctx.fillText(tickText, canvasWidth - CONTROL_BAR_PADDING, centerY);
    }

    /**
     * Render play/pause button.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Button rectangle
     * @param {boolean} playing - Current play state
     * @param {boolean} hovering - Whether mouse is hovering
     */
    function renderPlayPauseButton(ctx, rect, playing, hovering) {
        const color = hovering ? BUTTON_HOVER_COLOR : BUTTON_COLOR;
        const centerX = rect.x + rect.width / 2;
        const centerY = rect.y + rect.height / 2;
        const size = rect.width * 0.4;

        ctx.fillStyle = color;
        ctx.beginPath();

        if (playing) {
            // Pause icon (two vertical bars)
            const barWidth = size * 0.3;
            const barHeight = size * 1.2;
            const gap = size * 0.3;

            ctx.fillRect(
                centerX - gap / 2 - barWidth,
                centerY - barHeight / 2,
                barWidth,
                barHeight
            );
            ctx.fillRect(
                centerX + gap / 2,
                centerY - barHeight / 2,
                barWidth,
                barHeight
            );
        } else {
            // Play icon (triangle pointing right)
            ctx.moveTo(centerX - size * 0.4, centerY - size * 0.6);
            ctx.lineTo(centerX - size * 0.4, centerY + size * 0.6);
            ctx.lineTo(centerX + size * 0.6, centerY);
            ctx.closePath();
            ctx.fill();
        }
    }

    /**
     * Render speed slider.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Slider track rectangle
     * @param {number} speed - Current speed value
     * @param {number} centerY - Vertical center
     */
    function renderSpeedSlider(ctx, rect, speed, centerY) {
        // Calculate normalized position using log scale
        const logMin = Math.log(MIN_SPEED);
        const logMax = Math.log(MAX_SPEED);
        const logSpeed = Math.log(speed);
        const normalized = (logSpeed - logMin) / (logMax - logMin);

        // Track background
        ctx.fillStyle = SLIDER_TRACK_COLOR;
        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 3);
        ctx.fill();

        // Filled portion
        const fillWidth = normalized * rect.width;
        ctx.fillStyle = SLIDER_FILL_COLOR;
        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, fillWidth, rect.height, 3);
        ctx.fill();

        // Knob
        sliderKnobX = rect.x + fillWidth;
        ctx.fillStyle = SLIDER_KNOB_COLOR;
        ctx.beginPath();
        ctx.arc(sliderKnobX, centerY, SLIDER_KNOB_SIZE / 2, 0, Math.PI * 2);
        ctx.fill();

        // Knob border when hovering or dragging
        if (isHoveringSlider || isDraggingSlider) {
            ctx.strokeStyle = BUTTON_COLOR;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }

    /**
     * Check if a point is in the control bar (for interaction.js to know when to suppress other interactions).
     * @param {number} y - Y coordinate
     * @param {number} canvasHeight - Canvas height
     * @returns {boolean} True if in control bar area
     */
    function isPointInControlBar(y, canvasHeight) {
        return y >= canvasHeight - CONTROL_BAR_HEIGHT;
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineControls = {
        // Initialization
        init: init,
        destroy: destroy,

        // State updates
        updateTick: updateTick,
        setPlaying: setPlaying,
        setSpeed: setSpeed,

        // Getters
        isPlaying: function() { return isPlaying; },
        getSpeed: function() { return currentSpeed; },
        getTick: function() { return currentTick; },
        isPointInControlBar: isPointInControlBar,

        // Rendering
        renderControlBar: renderControlBar,

        // Constants
        CONTROL_BAR_HEIGHT: CONTROL_BAR_HEIGHT,
        MIN_SPEED: MIN_SPEED,
        MAX_SPEED: MAX_SPEED
    };

})(typeof window !== 'undefined' ? window : this);
