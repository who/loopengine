/**
 * LoopEngine Visualizer - Main JavaScript
 *
 * Connects to the LoopEngine server via WebSocket and renders
 * the simulation frame data on an HTML5 Canvas.
 *
 * Uses LoopEngineRenderer for frame interpolation and layer-ordered rendering.
 */

(function() {
    'use strict';

    // Configuration
    const WS_FRAMES_URL = 'ws://localhost:8000/ws/frames';
    const WS_CONTROL_URL = 'ws://localhost:8000/ws/control';
    const RECONNECT_DELAY_MS = 3000;
    const MAX_RECONNECT_DELAY_MS = 30000;  // Max delay between reconnection attempts
    const SERVER_FRAME_RATE = 30;  // Expected server frame rate for interpolation

    // State
    let canvas = null;
    let ctx = null;
    let framesSocket = null;
    let controlSocket = null;
    let latestFrame = null;
    let animationFrameId = null;
    let connected = false;
    let animationTime = 0;
    let lastTimestamp = 0;

    // Connection state tracking
    let framesReconnectAttempts = 0;
    let controlReconnectAttempts = 0;
    let connectionStatus = 'connecting';  // 'connected', 'connecting', 'reconnecting', 'error'

    /**
     * Initialize the canvas and start the application.
     */
    function init() {
        canvas = document.getElementById('canvas');
        if (!canvas) {
            console.error('Canvas element not found');
            return;
        }

        ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error('Could not get 2D context');
            return;
        }

        // Handle window resize
        window.addEventListener('resize', handleResize);
        handleResize();

        // Initialize interaction module for hover/click handling
        if (typeof LoopEngineInteraction !== 'undefined') {
            LoopEngineInteraction.init(canvas);
        }

        // Initialize controls module
        if (typeof LoopEngineControls !== 'undefined') {
            LoopEngineControls.init(canvas, sendControlCommand);
        }

        // Initialize GA module
        if (typeof LoopEngineGA !== 'undefined') {
            LoopEngineGA.init(canvas, sendControlCommand);
        }

        // Initialize Discovery module
        if (typeof LoopEngineDiscovery !== 'undefined') {
            LoopEngineDiscovery.init(canvas);
        }

        // Initialize Corpus selector module
        if (typeof LoopEngineCorpus !== 'undefined') {
            LoopEngineCorpus.init(canvas);
        }

        // Connect to WebSockets
        connectFramesSocket();
        connectControlSocket();

        // Start render loop
        startRenderLoop();

        console.log('LoopEngine Visualizer initialized');
    }

    /**
     * Handle window resize - update canvas dimensions.
     */
    function handleResize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        ctx.scale(dpr, dpr);

        // Store logical dimensions
        canvas._logicalWidth = rect.width;
        canvas._logicalHeight = rect.height;
    }

    /**
     * Calculate reconnection delay with exponential backoff.
     * @param {number} attempts - Number of reconnection attempts
     * @returns {number} Delay in milliseconds
     */
    function getReconnectDelay(attempts) {
        const delay = Math.min(
            RECONNECT_DELAY_MS * Math.pow(1.5, attempts),
            MAX_RECONNECT_DELAY_MS
        );
        return delay;
    }

    /**
     * Connect to the frames WebSocket endpoint.
     */
    function connectFramesSocket() {
        console.log('Connecting to frames WebSocket:', WS_FRAMES_URL);
        connectionStatus = framesReconnectAttempts > 0 ? 'reconnecting' : 'connecting';

        try {
            framesSocket = new WebSocket(WS_FRAMES_URL);

            framesSocket.onopen = function() {
                console.log('Frames WebSocket connected');
                connected = true;
                connectionStatus = 'connected';
                framesReconnectAttempts = 0;  // Reset on successful connection
            };

            framesSocket.onmessage = function(event) {
                try {
                    const frame = JSON.parse(event.data);
                    latestFrame = frame;

                    // Push frame to renderer for interpolation
                    if (typeof LoopEngineRenderer !== 'undefined') {
                        LoopEngineRenderer.pushFrame(frame);
                    }

                    // Only log every 100th frame to reduce console spam
                    if (frame.tick % 100 === 0) {
                        console.log('Frame received:', 'tick=' + frame.tick,
                                    'agents=' + frame.agents.length,
                                    'links=' + frame.links.length,
                                    'particles=' + frame.particles.length);
                    }
                } catch (e) {
                    console.error('Error parsing frame data:', e);
                }
            };

            framesSocket.onerror = function(error) {
                console.error('Frames WebSocket error:', error);
                connectionStatus = 'error';
            };

            framesSocket.onclose = function(event) {
                console.log('Frames WebSocket closed:', event.code, event.reason);
                connected = false;
                framesReconnectAttempts++;
                connectionStatus = 'reconnecting';

                // Attempt to reconnect with exponential backoff
                const delay = getReconnectDelay(framesReconnectAttempts);
                console.log('Reconnecting frames WebSocket in', delay, 'ms (attempt', framesReconnectAttempts, ')');
                setTimeout(connectFramesSocket, delay);
            };
        } catch (e) {
            console.error('Failed to create frames WebSocket:', e);
            framesReconnectAttempts++;
            connectionStatus = 'error';
            const delay = getReconnectDelay(framesReconnectAttempts);
            setTimeout(connectFramesSocket, delay);
        }
    }

    /**
     * Connect to the control WebSocket endpoint.
     */
    function connectControlSocket() {
        console.log('Connecting to control WebSocket:', WS_CONTROL_URL);

        try {
            controlSocket = new WebSocket(WS_CONTROL_URL);

            controlSocket.onopen = function() {
                console.log('Control WebSocket connected');
                controlReconnectAttempts = 0;  // Reset on successful connection
            };

            controlSocket.onmessage = function(event) {
                try {
                    const response = JSON.parse(event.data);
                    console.log('Control response:', response);

                    // Handle error responses
                    if (response.success === false) {
                        console.warn('Control command failed:', response.message);
                    }

                    // Route GA-related messages to GA module
                    if (typeof LoopEngineGA !== 'undefined') {
                        if (response.type === 'ga_progress') {
                            LoopEngineGA.handleGAProgress(response);
                        } else if (response.type === 'ga_complete') {
                            LoopEngineGA.handleGAComplete(response);
                        } else if (response.job_id !== undefined && response.status !== undefined) {
                            // This is a GA status response
                            LoopEngineGA.handleGAStatus(response);
                        } else if (response.job_id !== undefined && response.success !== undefined) {
                            // This could be start_ga or stop_ga response
                            if (response.message && response.message.includes('started')) {
                                LoopEngineGA.handleStartGAResponse(response);
                            } else if (response.message && response.message.includes('Stop')) {
                                LoopEngineGA.handleStopGAResponse(response);
                            }
                        }
                    }
                } catch (e) {
                    console.error('Error parsing control response:', e);
                }
            };

            controlSocket.onerror = function(error) {
                console.error('Control WebSocket error:', error);
            };

            controlSocket.onclose = function(event) {
                console.log('Control WebSocket closed:', event.code, event.reason);
                controlReconnectAttempts++;

                // Attempt to reconnect with exponential backoff
                const delay = getReconnectDelay(controlReconnectAttempts);
                console.log('Reconnecting control WebSocket in', delay, 'ms (attempt', controlReconnectAttempts, ')');
                setTimeout(connectControlSocket, delay);
            };
        } catch (e) {
            console.error('Failed to create control WebSocket:', e);
            controlReconnectAttempts++;
            const delay = getReconnectDelay(controlReconnectAttempts);
            setTimeout(connectControlSocket, delay);
        }
    }

    /**
     * Send a control command to the server.
     * @param {string} type - Command type: 'play', 'pause', 'set_speed', 'reset'
     * @param {Object} params - Additional parameters (e.g., {speed: 2.0})
     * @returns {boolean} True if command was sent, false if connection unavailable
     */
    function sendControlCommand(type, params) {
        if (!controlSocket || controlSocket.readyState !== WebSocket.OPEN) {
            console.warn('Control WebSocket not connected - command will be dropped:', type);
            return false;
        }

        try {
            const command = { type: type, ...params };
            controlSocket.send(JSON.stringify(command));
            console.log('Sent control command:', command);
            return true;
        } catch (e) {
            console.error('Failed to send control command:', type, e);
            return false;
        }
    }

    /**
     * Start the requestAnimationFrame render loop.
     */
    function startRenderLoop() {
        function loop(timestamp) {
            // Calculate delta time for animations
            if (lastTimestamp === 0) {
                lastTimestamp = timestamp;
            }
            const deltaTime = (timestamp - lastTimestamp) / 1000;
            lastTimestamp = timestamp;
            animationTime += deltaTime;

            render();
            animationFrameId = requestAnimationFrame(loop);
        }
        animationFrameId = requestAnimationFrame(loop);
    }

    /**
     * Main render function - called every animation frame.
     */
    function render() {
        const width = canvas._logicalWidth || canvas.width;
        const height = canvas._logicalHeight || canvas.height;

        // Use LoopEngineRenderer if available for interpolated rendering
        if (typeof LoopEngineRenderer !== 'undefined') {
            LoopEngineRenderer.render(
                ctx,
                performance.now(),
                animationTime,
                width,
                height,
                connected,
                SERVER_FRAME_RATE
            );
            return;
        }

        // Fallback: direct rendering without interpolation
        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        // Draw connection status
        if (!connected) {
            ctx.fillStyle = '#ff6b6b';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Connecting to server...', width / 2, height / 2);
            return;
        }

        // Draw frame if available
        if (latestFrame && latestFrame.tick >= 0) {
            drawFrame(latestFrame, width, height);
        } else {
            ctx.fillStyle = '#4a90d9';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for simulation data...', width / 2, height / 2);
        }
    }

    /**
     * Draw the complete frame.
     * @param {Object} frame - Frame data from the server
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     */
    function drawFrame(frame, width, height) {
        // Update viewport to center agents if not yet positioned
        if (viewport.offsetX === 0 && viewport.offsetY === 0 && frame.agents.length > 0) {
            // Calculate bounding box of all agents
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const agent of frame.agents) {
                minX = Math.min(minX, agent.x);
                minY = Math.min(minY, agent.y);
                maxX = Math.max(maxX, agent.x);
                maxY = Math.max(maxY, agent.y);
            }

            // Center the viewport on agents
            const contentWidth = maxX - minX + 100;
            const contentHeight = maxY - minY + 100;

            // Scale to fit with padding
            viewport.scale = Math.min(
                (width - 100) / contentWidth,
                (height - 100) / contentHeight,
                2.0  // Max scale
            );
            viewport.scale = Math.max(viewport.scale, 0.25);  // Min scale

            // Center offset
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            viewport.offsetX = width / 2 - centerX * viewport.scale;
            viewport.offsetY = height / 2 - centerY * viewport.scale;
        }

        // Draw label regions using the labels module (lowest layer - soft translucent clouds)
        if (typeof LoopEngineLabels !== 'undefined' && frame.label_regions) {
            LoopEngineLabels.renderLabelRegions(ctx, frame.label_regions, animationTime, viewport);
        }

        // Draw links using the links module (underneath agents and particles)
        if (typeof LoopEngineLinks !== 'undefined') {
            LoopEngineLinks.renderLinks(ctx, frame.links, animationTime, viewport);
        }

        // Draw particles using the particles module (on top of links, below agents)
        if (typeof LoopEngineParticles !== 'undefined') {
            LoopEngineParticles.renderParticles(ctx, frame.particles, frame.links, animationTime, viewport);
        }

        // Draw agents using the agents module (on top of everything)
        if (typeof LoopEngineAgents !== 'undefined') {
            LoopEngineAgents.renderAgents(ctx, frame.agents, animationTime, viewport);
        }

        // Draw tick/time info (overlay)
        ctx.fillStyle = '#666666';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Tick: ' + frame.tick + ' | Time: ' + frame.time.toFixed(2) + 's', 10, 20);

        // Draw stats
        ctx.fillText(
            'Agents: ' + frame.agents.length +
            ' | Links: ' + frame.links.length +
            ' | Particles: ' + frame.particles.length,
            10, 36
        );
    }

    // Expose control functions globally for debugging
    window.loopEngine = {
        play: function() { sendControlCommand('play'); },
        pause: function() { sendControlCommand('pause'); },
        setSpeed: function(speed) { sendControlCommand('set_speed', { speed: speed }); },
        reset: function() { sendControlCommand('reset'); },
        getLatestFrame: function() { return latestFrame; }
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
