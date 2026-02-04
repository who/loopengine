/**
 * LoopEngine Corpus Selector Module
 *
 * Provides a dropdown UI to switch between available corpora/scenarios.
 * Rendered in the control bar area, showing current corpus and allowing selection.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================

    const DROPDOWN_WIDTH = 160;
    const DROPDOWN_HEIGHT = 28;
    const DROPDOWN_PADDING = 8;
    const ITEM_HEIGHT = 32;
    const API_BASE_URL = 'http://localhost:8000';

    // Colors
    const DROPDOWN_BG_COLOR = 'rgba(40, 40, 60, 0.95)';
    const DROPDOWN_BORDER_COLOR = 'rgba(100, 100, 120, 0.5)';
    const DROPDOWN_HOVER_COLOR = 'rgba(60, 60, 80, 0.95)';
    const TEXT_COLOR = '#cccccc';
    const LABEL_COLOR = '#888888';
    const ITEM_HOVER_COLOR = 'rgba(74, 144, 217, 0.3)';
    const SELECTED_COLOR = '#4a90d9';
    const ARROW_COLOR = '#888888';

    // =========================================================================
    // State
    // =========================================================================

    let canvas = null;
    let isDropdownOpen = false;
    let isHoveringDropdown = false;
    let hoveredItemIndex = -1;

    // Corpus data
    let availableCorpora = [];
    let currentCorpusId = 'sandwich_shop';
    let isLoading = false;

    // UI element positions (calculated during render)
    let dropdownRect = { x: 0, y: 0, width: DROPDOWN_WIDTH, height: DROPDOWN_HEIGHT };
    let dropdownListRect = { x: 0, y: 0, width: DROPDOWN_WIDTH, height: 0 };

    // =========================================================================
    // Initialization
    // =========================================================================

    /**
     * Initialize the corpus selector module.
     * @param {HTMLCanvasElement} canvasElement - The canvas element
     */
    function init(canvasElement) {
        canvas = canvasElement;

        // Add event listeners
        canvas.addEventListener('mousedown', handleMouseDown);
        canvas.addEventListener('mousemove', handleMouseMove);

        // Fetch available corpora from server
        fetchCorpora();
    }

    /**
     * Cleanup event listeners.
     */
    function destroy() {
        if (canvas) {
            canvas.removeEventListener('mousedown', handleMouseDown);
            canvas.removeEventListener('mousemove', handleMouseMove);
        }
        canvas = null;
    }

    // =========================================================================
    // API Communication
    // =========================================================================

    /**
     * Fetch available corpora from the server.
     */
    async function fetchCorpora() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/corpora`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            availableCorpora = data.corpora || [];
            currentCorpusId = data.current || 'sandwich_shop';
            console.log('Fetched corpora:', availableCorpora, 'current:', currentCorpusId);
        } catch (error) {
            console.error('Failed to fetch corpora:', error);
            // Use defaults if fetch fails
            availableCorpora = [
                { id: 'sandwich_shop', name: 'Sandwich Shop' },
                { id: 'software_team', name: 'Software Team' }
            ];
        }
    }

    /**
     * Load a corpus by ID.
     * @param {string} corpusId - Corpus ID to load
     */
    async function loadCorpus(corpusId) {
        if (isLoading || corpusId === currentCorpusId) {
            isDropdownOpen = false;
            return;
        }

        isLoading = true;
        console.log('Loading corpus:', corpusId);

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/world/load_corpus?corpus_name=${encodeURIComponent(corpusId)}`,
                { method: 'POST' }
            );

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            console.log('Corpus loaded:', result);
            currentCorpusId = corpusId;
        } catch (error) {
            console.error('Failed to load corpus:', error);
        } finally {
            isLoading = false;
            isDropdownOpen = false;
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

        // Check if clicking on dropdown button
        if (isInsideRect(pos.x, pos.y, dropdownRect)) {
            isDropdownOpen = !isDropdownOpen;
            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // Check if clicking on dropdown list items
        if (isDropdownOpen && isInsideRect(pos.x, pos.y, dropdownListRect)) {
            // Determine which item was clicked
            const relativeY = pos.y - dropdownListRect.y;
            const itemIndex = Math.floor(relativeY / ITEM_HEIGHT);

            if (itemIndex >= 0 && itemIndex < availableCorpora.length) {
                const corpus = availableCorpora[itemIndex];
                loadCorpus(corpus.id);
            }

            event.preventDefault();
            event.stopPropagation();
            return;
        }

        // Close dropdown if clicking elsewhere
        if (isDropdownOpen) {
            isDropdownOpen = false;
        }
    }

    /**
     * Handle mouse move event.
     * @param {MouseEvent} event - Mouse event
     */
    function handleMouseMove(event) {
        const pos = getMousePos(event);

        // Check hover over dropdown button
        isHoveringDropdown = isInsideRect(pos.x, pos.y, dropdownRect);

        // Check hover over dropdown items
        if (isDropdownOpen && isInsideRect(pos.x, pos.y, dropdownListRect)) {
            const relativeY = pos.y - dropdownListRect.y;
            hoveredItemIndex = Math.floor(relativeY / ITEM_HEIGHT);

            if (hoveredItemIndex < 0 || hoveredItemIndex >= availableCorpora.length) {
                hoveredItemIndex = -1;
            }
        } else {
            hoveredItemIndex = -1;
        }

        // Update cursor
        if (isHoveringDropdown || hoveredItemIndex >= 0) {
            canvas.style.cursor = 'pointer';
        }
    }

    // =========================================================================
    // Rendering
    // =========================================================================

    /**
     * Get the display name for the current corpus.
     * @returns {string} Display name
     */
    function getCurrentCorpusName() {
        const corpus = availableCorpora.find(c => c.id === currentCorpusId);
        return corpus ? corpus.name : currentCorpusId;
    }

    /**
     * Render the corpus selector dropdown.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} canvasWidth - Canvas width
     * @param {number} canvasHeight - Canvas height
     */
    function renderCorpusSelector(ctx, canvasWidth, canvasHeight) {
        // Position dropdown in the control bar area, after the speed slider area
        // Control bar is at bottom, we'll position this to the right of the speed controls
        const CONTROL_BAR_HEIGHT = 48;
        const barY = canvasHeight - CONTROL_BAR_HEIGHT;
        const centerY = barY + CONTROL_BAR_HEIGHT / 2;

        // Position to the right, leaving space for tick counter
        const rightMargin = 140; // Space for tick counter
        const xPos = canvasWidth - rightMargin - DROPDOWN_WIDTH - 16;

        // Update dropdown rect
        dropdownRect = {
            x: xPos,
            y: centerY - DROPDOWN_HEIGHT / 2,
            width: DROPDOWN_WIDTH,
            height: DROPDOWN_HEIGHT
        };

        // Draw dropdown button
        const bgColor = isHoveringDropdown || isDropdownOpen
            ? DROPDOWN_HOVER_COLOR
            : DROPDOWN_BG_COLOR;

        ctx.fillStyle = bgColor;
        ctx.beginPath();
        ctx.roundRect(dropdownRect.x, dropdownRect.y, dropdownRect.width, dropdownRect.height, 4);
        ctx.fill();

        ctx.strokeStyle = DROPDOWN_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw current corpus name
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';

        const textX = dropdownRect.x + DROPDOWN_PADDING;
        const textY = dropdownRect.y + dropdownRect.height / 2;
        const displayName = isLoading ? 'Loading...' : getCurrentCorpusName();

        // Clip text if too long
        const maxTextWidth = DROPDOWN_WIDTH - DROPDOWN_PADDING * 2 - 16; // Leave space for arrow
        ctx.save();
        ctx.beginPath();
        ctx.rect(textX, dropdownRect.y, maxTextWidth, dropdownRect.height);
        ctx.clip();
        ctx.fillText(displayName, textX, textY);
        ctx.restore();

        // Draw dropdown arrow
        const arrowX = dropdownRect.x + dropdownRect.width - 16;
        const arrowY = textY;
        ctx.fillStyle = ARROW_COLOR;
        ctx.beginPath();
        if (isDropdownOpen) {
            // Up arrow
            ctx.moveTo(arrowX, arrowY + 3);
            ctx.lineTo(arrowX + 6, arrowY + 3);
            ctx.lineTo(arrowX + 3, arrowY - 3);
        } else {
            // Down arrow
            ctx.moveTo(arrowX, arrowY - 3);
            ctx.lineTo(arrowX + 6, arrowY - 3);
            ctx.lineTo(arrowX + 3, arrowY + 3);
        }
        ctx.closePath();
        ctx.fill();

        // Draw dropdown list if open
        if (isDropdownOpen && availableCorpora.length > 0) {
            renderDropdownList(ctx);
        }
    }

    /**
     * Render the dropdown list.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    function renderDropdownList(ctx) {
        const listHeight = availableCorpora.length * ITEM_HEIGHT;

        // Position above the dropdown button
        dropdownListRect = {
            x: dropdownRect.x,
            y: dropdownRect.y - listHeight - 4,
            width: DROPDOWN_WIDTH,
            height: listHeight
        };

        // Draw list background
        ctx.fillStyle = DROPDOWN_BG_COLOR;
        ctx.beginPath();
        ctx.roundRect(dropdownListRect.x, dropdownListRect.y, dropdownListRect.width, dropdownListRect.height, 4);
        ctx.fill();

        ctx.strokeStyle = DROPDOWN_BORDER_COLOR;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw items
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < availableCorpora.length; i++) {
            const corpus = availableCorpora[i];
            const itemY = dropdownListRect.y + i * ITEM_HEIGHT;
            const isHovered = i === hoveredItemIndex;
            const isSelected = corpus.id === currentCorpusId;

            // Draw item background
            if (isHovered) {
                ctx.fillStyle = ITEM_HOVER_COLOR;
                ctx.fillRect(dropdownListRect.x + 2, itemY + 2, DROPDOWN_WIDTH - 4, ITEM_HEIGHT - 4);
            }

            // Draw item text
            ctx.fillStyle = isSelected ? SELECTED_COLOR : TEXT_COLOR;
            const textX = dropdownListRect.x + DROPDOWN_PADDING;
            const textY = itemY + ITEM_HEIGHT / 2;
            ctx.fillText(corpus.name, textX, textY);

            // Draw check mark for selected item
            if (isSelected) {
                ctx.fillStyle = SELECTED_COLOR;
                const checkX = dropdownListRect.x + DROPDOWN_WIDTH - 20;
                ctx.fillText('✓', checkX, textY);
            }
        }
    }

    /**
     * Check if a point is in the corpus selector area.
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {boolean} True if in corpus selector area
     */
    function isPointInCorpusSelector(x, y) {
        if (isInsideRect(x, y, dropdownRect)) {
            return true;
        }
        if (isDropdownOpen && isInsideRect(x, y, dropdownListRect)) {
            return true;
        }
        return false;
    }

    /**
     * Get the current corpus ID.
     * @returns {string} Current corpus ID
     */
    function getCurrentCorpus() {
        return currentCorpusId;
    }

    /**
     * Check if the dropdown is open.
     * @returns {boolean} True if dropdown is open
     */
    function isOpen() {
        return isDropdownOpen;
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineCorpus = {
        // Initialization
        init: init,
        destroy: destroy,

        // State
        getCurrentCorpus: getCurrentCorpus,
        isOpen: isOpen,
        isPointInCorpusSelector: isPointInCorpusSelector,

        // API
        loadCorpus: loadCorpus,
        fetchCorpora: fetchCorpora,

        // Rendering
        renderCorpusSelector: renderCorpusSelector
    };

})(typeof window !== 'undefined' ? window : this);
