/**
 * LoopEngine Discovery Module
 *
 * Provides genome discovery UI with trigger button, schema display,
 * diff view, and flexibility display per role.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================

    const PANEL_WIDTH = 320;
    const PANEL_PADDING = 16;
    const POLL_INTERVAL_MS = 1000;

    // Colors
    const PANEL_BG_COLOR = 'rgba(26, 26, 46, 0.95)';
    const PANEL_BORDER_COLOR = 'rgba(100, 100, 120, 0.5)';
    const TEXT_COLOR = '#cccccc';
    const LABEL_COLOR = '#888888';
    const BUTTON_COLOR = '#4a90d9';
    const BUTTON_HOVER_COLOR = '#5da0e9';
    const BUTTON_DISABLED_COLOR = '#555555';
    const SUCCESS_COLOR = '#4ad97a';
    const ERROR_COLOR = '#d94a4a';
    const DIFF_ADD_COLOR = 'rgba(74, 217, 122, 0.3)';
    const DIFF_REMOVE_COLOR = 'rgba(217, 74, 74, 0.3)';
    const CATEGORY_COLORS = {
        'cognitive': '#9b59b6',
        'physical': '#e74c3c',
        'social': '#3498db',
        'temporal': '#f39c12',
        'default': '#95a5a6'
    };

    // =========================================================================
    // State
    // =========================================================================

    let canvas = null;
    let panelVisible = true;

    // Discovery state
    let isDiscovering = false;
    let currentJobId = null;
    let pollIntervalId = null;
    let lastError = null;
    let discoveryStatus = 'idle'; // idle, running, completed, failed

    // Schema state
    let currentSchemas = {};  // role -> schema
    let previousSchemas = {}; // for diff view
    let showDiff = false;
    let migratedAgentCount = 0;

    // UI interaction state
    let isHoveringDiscover = false;
    let isHoveringToggle = false;
    let scrollOffset = 0;
    let maxScrollOffset = 0;
    let isDraggingScroll = false;

    // UI element positions
    let panelRect = { x: 0, y: 0, width: PANEL_WIDTH, height: 0 };
    let discoverButtonRect = { x: 0, y: 0, width: 0, height: 0 };
    let toggleButtonRect = { x: 0, y: 0, width: 24, height: 24 };

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * Initialize the discovery module.
     * @param {HTMLCanvasElement} canvasElement - The canvas element
     */
    function init(canvasElement) {
        canvas = canvasElement;

        // Add event listeners
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('mouseup', handleMouseUp);
        canvas.addEventListener('wheel', handleWheel);

        // Load initial schemas
        fetchSchemas();
    }

    /**
     * Cleanup event listeners.
     */
    function destroy() {
        if (canvas) {
            canvas.removeEventListener('mousedown', handleMouseDown);
            canvas.removeEventListener('mousemove', handleMouseMove);
            canvas.removeEventListener('mouseup', handleMouseUp);
            canvas.removeEventListener('wheel', handleWheel);
        }
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
        }
        canvas = null;
    }

    // =========================================================================
    // API Functions
    // =========================================================================

    /**
     * Fetch current schemas from API.
     */
    async function fetchSchemas() {
        try {
            const response = await fetch('/api/schemas');
            if (response.ok) {
                const schemas = await response.json();
                // Convert array to role -> schema map
                const schemaMap = {};
                for (const schema of schemas) {
                    schemaMap[schema.role] = schema;
                }
                currentSchemas = schemaMap;
            }
        } catch (e) {
            console.error('Failed to fetch schemas:', e);
        }
    }

    /**
     * Start a discovery run via REST API.
     */
    async function startDiscovery() {
        if (isDiscovering) return;

        isDiscovering = true;
        discoveryStatus = 'running';
        lastError = null;
        showDiff = false;

        // Store current schemas for diff
        previousSchemas = JSON.parse(JSON.stringify(currentSchemas));

        try {
            // Get system description from world state
            const systemDescription = await getSystemDescription();

            const response = await fetch('/api/discovery/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(systemDescription)
            });

            if (response.ok) {
                const data = await response.json();
                currentJobId = data.job_id;
                startPolling();
            } else {
                const error = await response.json();
                lastError = error.detail || 'Failed to start discovery';
                discoveryStatus = 'failed';
                isDiscovering = false;
            }
        } catch (e) {
            lastError = e.message || 'Network error';
            discoveryStatus = 'failed';
            isDiscovering = false;
        }
    }

    /**
     * Get system description for discovery API.
     */
    async function getSystemDescription() {
        // Fetch agents to determine roles
        try {
            const response = await fetch('/api/agents');
            if (response.ok) {
                const agents = await response.json();
                const roles = {};

                // Group agents by role and build role descriptions
                for (const agent of agents) {
                    if (!roles[agent.role]) {
                        roles[agent.role] = {
                            name: agent.role,
                            inputs: [],
                            outputs: [],
                            constraints: [],
                            links_to: []
                        };
                    }
                }

                // Build description based on roles found
                return {
                    system: 'Simulation system with ' + Object.keys(roles).length + ' roles',
                    roles: Object.values(roles)
                };
            }
        } catch (e) {
            console.error('Failed to get agents:', e);
        }

        // Fallback to default sandwich shop description
        return {
            system: 'A sandwich shop with workers serving customers',
            roles: [
                {
                    name: 'sandwich_maker',
                    inputs: ['customer orders', 'ingredients inventory'],
                    outputs: ['completed sandwiches'],
                    constraints: ['speed matters', 'quality consistency'],
                    links_to: ['cashier (service)']
                },
                {
                    name: 'cashier',
                    inputs: ['customers', 'payments'],
                    outputs: ['receipts', 'orders to kitchen'],
                    constraints: ['fast checkout', 'accuracy'],
                    links_to: ['sandwich_maker (service)']
                },
                {
                    name: 'owner',
                    inputs: ['financial reports', 'customer feedback'],
                    outputs: ['pricing decisions', 'staff schedules'],
                    constraints: ['profitability', 'staff satisfaction'],
                    links_to: ['sandwich_maker (hierarchical)', 'cashier (hierarchical)']
                }
            ]
        };
    }

    /**
     * Start polling for discovery status.
     */
    function startPolling() {
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
        }
        pollIntervalId = setInterval(pollDiscoveryStatus, POLL_INTERVAL_MS);
    }

    /**
     * Poll discovery status.
     */
    async function pollDiscoveryStatus() {
        if (!currentJobId) return;

        try {
            const response = await fetch(`/api/discovery/status/${currentJobId}`);
            if (response.ok) {
                const data = await response.json();

                if (data.status === 'completed') {
                    stopPolling();
                    discoveryStatus = 'completed';
                    isDiscovering = false;
                    migratedAgentCount = data.migrated_agent_count || 0;

                    // Update schemas
                    if (data.discovered_schemas) {
                        const schemaMap = {};
                        for (const [role, schema] of Object.entries(data.discovered_schemas)) {
                            schemaMap[role] = {
                                role: role,
                                traits: schema.traits ? Object.values(schema.traits) : [],
                                flexibility_score: schema.flexibility_score || 0.5
                            };
                        }
                        currentSchemas = schemaMap;
                        showDiff = Object.keys(previousSchemas).length > 0;
                    }
                } else if (data.status === 'failed') {
                    stopPolling();
                    discoveryStatus = 'failed';
                    isDiscovering = false;
                    lastError = data.error_message || 'Discovery failed';
                }
                // Still running - continue polling
            }
        } catch (e) {
            console.error('Failed to poll discovery status:', e);
        }
    }

    /**
     * Stop polling.
     */
    function stopPolling() {
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
        }
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

        // Check discover button
        if (!isDiscovering && isInsideRect(pos.x, pos.y, discoverButtonRect)) {
            startDiscovery();
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
            isHoveringDiscover = false;
            return;
        }

        if (!isInsideRect(pos.x, pos.y, panelRect)) {
            isHoveringDiscover = false;
            return;
        }

        isHoveringDiscover = !isDiscovering && isInsideRect(pos.x, pos.y, discoverButtonRect);

        // Update cursor
        if (isHoveringDiscover || isHoveringToggle) {
            canvas.style.cursor = 'pointer';
        }
    }

    /**
     * Handle mouse up event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseUp(event) {
        isDraggingScroll = false;
    }

    /**
     * Handle mouse wheel for scrolling.
     * @param {WheelEvent} event - Wheel event
     */
    function handleWheel(event) {
        if (!panelVisible) return;

        const pos = getMousePos(event);
        if (!isInsideRect(pos.x, pos.y, panelRect)) return;

        scrollOffset = Math.max(0, Math.min(maxScrollOffset, scrollOffset + event.deltaY * 0.5));
        event.preventDefault();
    }

    // =========================================================================
    // Rendering
    // =========================================================================

    /**
     * Render the discovery panel.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     */
    function renderDiscoveryPanel(ctx, canvasWidth, canvasHeight) {
        // Render toggle button (always visible)
        renderToggleButton(ctx);

        if (!panelVisible) return;

        // Calculate panel position (top-left, below status info)
        const panelX = 12;
        const panelY = 60;
        const maxPanelHeight = canvasHeight - 130; // Leave room for control bar

        // Calculate content height
        let contentHeight = PANEL_PADDING; // Top padding

        // Title
        contentHeight += 24;
        contentHeight += 12;

        // Discover button
        contentHeight += 40;
        contentHeight += 12;

        // Status
        if (discoveryStatus !== 'idle') {
            contentHeight += 24;
            contentHeight += 8;
        }

        // Schemas
        const roles = Object.keys(currentSchemas);
        for (const role of roles) {
            contentHeight += 28; // Role header
            const schema = currentSchemas[role];
            const traits = schema.traits || [];
            contentHeight += traits.length * 24 + 8; // Traits
            contentHeight += 24; // Flexibility
            contentHeight += 16; // Spacing
        }

        if (roles.length === 0) {
            contentHeight += 48; // No schemas message
        }

        contentHeight += PANEL_PADDING; // Bottom padding

        // Limit panel height and enable scrolling
        const actualPanelHeight = Math.min(contentHeight, maxPanelHeight);
        maxScrollOffset = Math.max(0, contentHeight - actualPanelHeight);

        panelRect = { x: panelX, y: panelY, width: PANEL_WIDTH, height: actualPanelHeight };

        // Draw panel background
        ctx.fillStyle = PANEL_BG_COLOR;
        ctx.beginPath();
        ctx.roundRect(panelRect.x, panelRect.y, panelRect.width, panelRect.height, 8);
        ctx.fill();

        // Draw panel border
        ctx.strokeStyle = PANEL_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Set up clipping region for content
        ctx.save();
        ctx.beginPath();
        ctx.rect(panelRect.x, panelRect.y, panelRect.width, panelRect.height);
        ctx.clip();

        // Draw content
        let y = panelY + PANEL_PADDING - scrollOffset;
        const contentX = panelX + PANEL_PADDING;
        const contentWidth = PANEL_WIDTH - PANEL_PADDING * 2;

        // Title
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = 'bold 14px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('Genome Discovery', contentX, y);
        y += 24 + 12;

        // Discover button
        discoverButtonRect = { x: contentX, y: y, width: contentWidth, height: 36 };
        renderDiscoverButton(ctx, discoverButtonRect);
        y += 40 + 12;

        // Status
        if (discoveryStatus !== 'idle') {
            renderStatus(ctx, contentX, y, contentWidth);
            y += 24 + 8;
        }

        // Schemas
        y = renderSchemas(ctx, contentX, y, contentWidth, roles);

        ctx.restore();

        // Draw scroll indicator if needed
        if (maxScrollOffset > 0) {
            renderScrollIndicator(ctx, panelRect, scrollOffset, maxScrollOffset);
        }
    }

    /**
     * Render toggle button.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    function renderToggleButton(ctx) {
        toggleButtonRect = {
            x: 12,
            y: 36,
            width: 24,
            height: 24
        };

        ctx.fillStyle = isHoveringToggle ? BUTTON_HOVER_COLOR : BUTTON_COLOR;
        ctx.beginPath();
        ctx.roundRect(toggleButtonRect.x, toggleButtonRect.y, toggleButtonRect.width, toggleButtonRect.height, 4);
        ctx.fill();

        // Draw DNA icon
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('AI', toggleButtonRect.x + 12, toggleButtonRect.y + 12);
    }

    /**
     * Render discover button.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Button rectangle
     */
    function renderDiscoverButton(ctx, rect) {
        const enabled = !isDiscovering;
        let color, hoverColor;

        if (isDiscovering) {
            color = BUTTON_DISABLED_COLOR;
            hoverColor = BUTTON_DISABLED_COLOR;
        } else {
            color = BUTTON_COLOR;
            hoverColor = BUTTON_HOVER_COLOR;
        }

        ctx.fillStyle = enabled && isHoveringDiscover ? hoverColor : color;
        ctx.beginPath();
        ctx.roundRect(rect.x, rect.y, rect.width, rect.height, 4);
        ctx.fill();

        ctx.fillStyle = enabled ? '#ffffff' : '#888888';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        let label = 'Discover Genomes';
        if (isDiscovering) {
            // Animated dots
            const dots = '.'.repeat((Math.floor(Date.now() / 300) % 4));
            label = 'Discovering' + dots;
        }

        ctx.fillText(label, rect.x + rect.width / 2, rect.y + rect.height / 2);
    }

    /**
     * Render status message.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Available width
     */
    function renderStatus(ctx, x, y, width) {
        ctx.font = '11px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';

        if (discoveryStatus === 'running') {
            ctx.fillStyle = BUTTON_COLOR;
            ctx.fillText('Running discovery...', x, y);
        } else if (discoveryStatus === 'completed') {
            ctx.fillStyle = SUCCESS_COLOR;
            ctx.fillText('Discovery complete! ' + migratedAgentCount + ' agents migrated.', x, y);
        } else if (discoveryStatus === 'failed') {
            ctx.fillStyle = ERROR_COLOR;
            ctx.fillText('Error: ' + (lastError || 'Unknown error'), x, y);
        }
    }

    /**
     * Render schemas list.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Available width
     * @param {Array} roles - Array of role names
     * @returns {number} Final Y position
     */
    function renderSchemas(ctx, x, y, width, roles) {
        if (roles.length === 0) {
            ctx.fillStyle = LABEL_COLOR;
            ctx.font = '11px monospace';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('No schemas discovered yet.', x, y);
            ctx.fillText('Click "Discover Genomes" to start.', x, y + 16);
            return y + 48;
        }

        for (const role of roles) {
            const schema = currentSchemas[role];
            const prevSchema = previousSchemas[role];
            const isNew = showDiff && !prevSchema;

            // Role header
            ctx.fillStyle = TEXT_COLOR;
            ctx.font = 'bold 12px monospace';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';

            // Highlight new roles
            if (isNew) {
                ctx.fillStyle = DIFF_ADD_COLOR;
                ctx.fillRect(x - 4, y - 2, width + 8, 24);
                ctx.fillStyle = SUCCESS_COLOR;
            }

            ctx.fillText(formatRoleName(role), x, y);
            y += 28;

            // Traits
            const traits = schema.traits || [];
            for (const trait of traits) {
                const isNewTrait = showDiff && prevSchema &&
                    !prevSchema.traits.some(t => t.name === trait.name);
                const isRemovedTrait = false; // We'd need to check diff

                // Trait background for diff
                if (isNewTrait) {
                    ctx.fillStyle = DIFF_ADD_COLOR;
                    ctx.fillRect(x - 4, y - 2, width + 8, 22);
                }

                // Category color indicator
                const category = trait.category || 'default';
                const categoryColor = CATEGORY_COLORS[category] || CATEGORY_COLORS['default'];
                ctx.fillStyle = categoryColor;
                ctx.beginPath();
                ctx.arc(x + 6, y + 8, 4, 0, Math.PI * 2);
                ctx.fill();

                // Trait name and range
                ctx.fillStyle = TEXT_COLOR;
                ctx.font = '11px monospace';
                const traitText = trait.name + ' [' + trait.min_val.toFixed(1) + '-' + trait.max_val.toFixed(1) + ']';
                ctx.fillText(traitText, x + 16, y);

                y += 24;
            }

            // Flexibility score
            const flexibility = schema.flexibility_score !== undefined ? schema.flexibility_score : 0.5;
            y += 4;
            renderFlexibilityBar(ctx, x, y, width, flexibility);
            y += 24;
            y += 8;
        }

        // Show removed roles in diff view
        if (showDiff) {
            for (const role of Object.keys(previousSchemas)) {
                if (!currentSchemas[role]) {
                    ctx.fillStyle = DIFF_REMOVE_COLOR;
                    ctx.fillRect(x - 4, y - 2, width + 8, 24);
                    ctx.fillStyle = ERROR_COLOR;
                    ctx.font = 'bold 12px monospace';
                    ctx.fillText(formatRoleName(role) + ' (removed)', x, y);
                    y += 28;
                }
            }
        }

        return y;
    }

    /**
     * Render flexibility bar.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {number} width - Bar width
     * @param {number} value - Flexibility value (0-1)
     */
    function renderFlexibilityBar(ctx, x, y, width, value) {
        const barHeight = 6;
        const barWidth = width - 80;
        const barX = x + 60;

        // Label
        ctx.fillStyle = LABEL_COLOR;
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('Flex:', x, y - 1);

        // Background
        ctx.fillStyle = 'rgba(100, 100, 120, 0.3)';
        ctx.beginPath();
        ctx.roundRect(barX, y, barWidth, barHeight, 2);
        ctx.fill();

        // Fill
        const fillWidth = value * barWidth;
        const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
        gradient.addColorStop(0, '#3498db');
        gradient.addColorStop(0.5, '#9b59b6');
        gradient.addColorStop(1, '#e74c3c');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(barX, y, fillWidth, barHeight, 2);
        ctx.fill();

        // Value text
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText((value * 100).toFixed(0) + '%', barX + barWidth + 8, y - 1);
    }

    /**
     * Render scroll indicator.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} rect - Panel rectangle
     * @param {number} offset - Current scroll offset
     * @param {number} maxOffset - Maximum scroll offset
     */
    function renderScrollIndicator(ctx, rect, offset, maxOffset) {
        const trackHeight = rect.height - 16;
        const trackX = rect.x + rect.width - 8;
        const trackY = rect.y + 8;

        const thumbHeight = Math.max(20, trackHeight * (rect.height / (rect.height + maxOffset)));
        const thumbY = trackY + (offset / maxOffset) * (trackHeight - thumbHeight);

        // Track
        ctx.fillStyle = 'rgba(100, 100, 120, 0.2)';
        ctx.beginPath();
        ctx.roundRect(trackX, trackY, 4, trackHeight, 2);
        ctx.fill();

        // Thumb
        ctx.fillStyle = 'rgba(100, 100, 120, 0.5)';
        ctx.beginPath();
        ctx.roundRect(trackX, thumbY, 4, thumbHeight, 2);
        ctx.fill();
    }

    /**
     * Format role name for display.
     * @param {string} role - Role identifier
     * @returns {string} Formatted name
     */
    function formatRoleName(role) {
        return role.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    /**
     * Check if a point is in the discovery panel area.
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

    global.LoopEngineDiscovery = {
        // Initialization
        init: init,
        destroy: destroy,

        // State
        isDiscovering: function() { return isDiscovering; },
        getSchemas: function() { return currentSchemas; },
        isPointInPanel: isPointInPanel,

        // Rendering
        renderDiscoveryPanel: renderDiscoveryPanel,

        // Actions
        startDiscovery: startDiscovery,
        fetchSchemas: fetchSchemas
    };

})(typeof window !== 'undefined' ? window : this);
