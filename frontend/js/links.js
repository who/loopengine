/**
 * LoopEngine Link Rendering Module
 *
 * Renders links as swaying bezier curves with thickness encoding interaction
 * density and color encoding link type.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Link Type Colors (fallback if not provided by backend)
    // =========================================================================

    const LINK_TYPE_COLORS = {
        hierarchical: '#e74c3c',  // Red
        peer: '#3498db',          // Blue
        service: '#2ecc71',       // Green
        competitive: '#f39c12',   // Orange
        default: '#666666'        // Gray
    };

    // =========================================================================
    // Sway Animation Configuration
    // =========================================================================

    const SWAY_AMPLITUDE = 8;      // Pixels of perpendicular displacement
    const SWAY_SPEED = 0.5;        // Radians per second

    // Offset for bidirectional links (pixels apart)
    const BIDIRECTIONAL_OFFSET = 12;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /**
     * Calculate the perpendicular offset vector for a line segment.
     * @param {number} x1 - Start X
     * @param {number} y1 - Start Y
     * @param {number} x2 - End X
     * @param {number} y2 - End Y
     * @returns {Object} Normalized perpendicular vector {px, py}
     */
    function getPerpendicularVector(x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);

        if (length < 0.001) {
            return { px: 0, py: 1 };
        }

        // Normalized perpendicular (rotated 90 degrees)
        return {
            px: -dy / length,
            py: dx / length
        };
    }

    /**
     * Apply sway displacement to control points.
     * @param {Array} controlPoints - Array of [x, y] control points
     * @param {number} swayPhase - Current sway phase (0 to 2π)
     * @param {number} time - Animation time for additional wobble
     * @param {number} linkIdHash - Hash of link ID for unique animation offset
     * @returns {Array} Displaced control points
     */
    function applySwayToControlPoints(controlPoints, swayPhase, time, linkIdHash) {
        if (controlPoints.length !== 4) {
            return controlPoints;
        }

        const [start, ctrl1, ctrl2, end] = controlPoints;

        // Get perpendicular direction from line between endpoints
        const perp = getPerpendicularVector(start[0], start[1], end[0], end[1]);

        // Calculate sway offset using the provided phase plus time-based animation
        // Add linkIdHash offset for variety between links
        const phase = swayPhase + time * SWAY_SPEED + (linkIdHash % 100) * 0.1;
        const swayOffset = Math.sin(phase) * SWAY_AMPLITUDE;

        // Apply sway only to interior control points (not endpoints)
        // Control points at different phases for organic wave
        const ctrl1Offset = swayOffset * Math.sin(phase);
        const ctrl2Offset = swayOffset * Math.sin(phase + 0.5);

        return [
            start,
            [ctrl1[0] + perp.px * ctrl1Offset, ctrl1[1] + perp.py * ctrl1Offset],
            [ctrl2[0] + perp.px * ctrl2Offset, ctrl2[1] + perp.py * ctrl2Offset],
            end
        ];
    }

    /**
     * Compute a simple hash from a string for consistent random offsets.
     * @param {string} str - Input string
     * @returns {number} Hash value
     */
    function hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(i);
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    /**
     * Check if two links are bidirectional (A→B and B→A exist).
     * @param {Object} link - Current link
     * @param {Array} allLinks - All links in the frame
     * @returns {Object|null} The reverse link if found, null otherwise
     */
    function findReverseLink(link, allLinks) {
        return allLinks.find(other =>
            other.id !== link.id &&
            other.source_id === link.dest_id &&
            other.dest_id === link.source_id
        );
    }

    /**
     * Calculate offset direction for bidirectional links.
     * Uses consistent ordering so each direction is offset to its own side.
     * @param {Object} link - The link
     * @param {Object} reverseLink - The reverse link (if exists)
     * @returns {number} 1 for one side, -1 for the other
     */
    function getBidirectionalSide(link, reverseLink) {
        if (!reverseLink) return 0;

        // Use string comparison for consistent ordering
        return link.source_id < link.dest_id ? 1 : -1;
    }

    // =========================================================================
    // Drawing Functions
    // =========================================================================

    /**
     * Draw a single cubic bezier curve.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} controlPoints - Array of 4 [x, y] points
     * @param {string} color - Stroke color
     * @param {number} thickness - Line thickness
     * @param {number} alpha - Opacity (0-1)
     */
    function drawBezierCurve(ctx, controlPoints, color, thickness, alpha) {
        if (controlPoints.length !== 4) return;

        const [start, ctrl1, ctrl2, end] = controlPoints;

        ctx.save();

        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = thickness;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        ctx.moveTo(start[0], start[1]);
        ctx.bezierCurveTo(
            ctrl1[0], ctrl1[1],
            ctrl2[0], ctrl2[1],
            end[0], end[1]
        );
        ctx.stroke();

        ctx.restore();
    }

    /**
     * Draw a link glow effect (subtle outer glow).
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} controlPoints - Array of 4 [x, y] points
     * @param {string} color - Glow color
     * @param {number} thickness - Base line thickness
     */
    function drawLinkGlow(ctx, controlPoints, color, thickness) {
        // Draw a slightly thicker, more transparent line behind
        drawBezierCurve(ctx, controlPoints, color, thickness + 4, 0.15);
        drawBezierCurve(ctx, controlPoints, color, thickness + 2, 0.25);
    }

    /**
     * Offset control points perpendicular to the link direction.
     * Used for bidirectional link separation.
     * @param {Array} controlPoints - Original control points
     * @param {number} offset - Offset distance
     * @param {number} direction - 1 or -1 for side selection
     * @returns {Array} Offset control points
     */
    function offsetControlPoints(controlPoints, offset, direction) {
        if (controlPoints.length !== 4) return controlPoints;

        const [start, ctrl1, ctrl2, end] = controlPoints;

        // Get perpendicular for the overall link direction
        const perp = getPerpendicularVector(start[0], start[1], end[0], end[1]);
        const offsetX = perp.px * offset * direction;
        const offsetY = perp.py * offset * direction;

        return [
            [start[0] + offsetX, start[1] + offsetY],
            [ctrl1[0] + offsetX, ctrl1[1] + offsetY],
            [ctrl2[0] + offsetX, ctrl2[1] + offsetY],
            [end[0] + offsetX, end[1] + offsetY]
        ];
    }

    /**
     * Transform control points to screen coordinates.
     * @param {Array} controlPoints - Control points in world coordinates
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     * @returns {Array} Screen-space control points
     */
    function transformControlPoints(controlPoints, viewport) {
        const scale = viewport ? viewport.scale : 1;
        const offsetX = viewport ? viewport.offsetX : 0;
        const offsetY = viewport ? viewport.offsetY : 0;

        return controlPoints.map(([x, y]) => [
            x * scale + offsetX,
            y * scale + offsetY
        ]);
    }

    // =========================================================================
    // Main Rendering Functions
    // =========================================================================

    /**
     * Render a single link on the canvas.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} link - LinkVisual object from frame
     * @param {Array} allLinks - All links for bidirectional detection
     * @param {number} time - Current animation time
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     */
    function renderLink(ctx, link, allLinks, time, viewport) {
        // Skip if no control points
        if (!link.control_points || link.control_points.length !== 4) {
            return;
        }

        // Get link properties
        const color = link.color || LINK_TYPE_COLORS[link.link_type] || LINK_TYPE_COLORS.default;
        const thickness = link.thickness || 2.0;
        const swayPhase = link.sway_phase || 0;
        const linkIdHash = hashString(link.id);

        // Check for bidirectional link
        const reverseLink = findReverseLink(link, allLinks);
        const biSide = getBidirectionalSide(link, reverseLink);

        // Start with original control points
        let controlPoints = link.control_points;

        // Apply bidirectional offset if needed
        if (biSide !== 0) {
            controlPoints = offsetControlPoints(
                controlPoints,
                BIDIRECTIONAL_OFFSET / 2,
                biSide
            );
        }

        // Transform to screen coordinates
        controlPoints = transformControlPoints(controlPoints, viewport);

        // Apply sway animation
        controlPoints = applySwayToControlPoints(controlPoints, swayPhase, time, linkIdHash);

        // Scale thickness with viewport
        const scale = viewport ? viewport.scale : 1;
        const scaledThickness = Math.max(1, thickness * Math.sqrt(scale));

        // Draw glow first (behind)
        drawLinkGlow(ctx, controlPoints, color, scaledThickness);

        // Draw the main curve
        drawBezierCurve(ctx, controlPoints, color, scaledThickness, 0.8);
    }

    /**
     * Render all links in a frame.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} links - Array of LinkVisual objects
     * @param {number} time - Current animation time
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     */
    function renderLinks(ctx, links, time, viewport) {
        if (!links || links.length === 0) return;

        // Sort links by type for consistent layering (hierarchical on top)
        const sortOrder = { hierarchical: 2, peer: 1, service: 1, competitive: 0 };
        const sortedLinks = [...links].sort((a, b) => {
            return (sortOrder[a.link_type] || 0) - (sortOrder[b.link_type] || 0);
        });

        for (const link of sortedLinks) {
            renderLink(ctx, link, links, time, viewport);
        }
    }

    /**
     * Get the link type color.
     * @param {string} linkType - Link type string
     * @returns {string} Hex color
     */
    function getLinkTypeColor(linkType) {
        return LINK_TYPE_COLORS[linkType] || LINK_TYPE_COLORS.default;
    }

    /**
     * Calculate a point on the rendered bezier curve.
     * Used for positioning particles along links.
     * @param {Object} link - LinkVisual object
     * @param {number} t - Parameter from 0 to 1
     * @param {number} time - Animation time (for sway)
     * @param {Array} allLinks - All links (for bidirectional offset)
     * @param {Object} viewport - Viewport transform
     * @returns {Object} Point {x, y} on the curve
     */
    function getPointOnLink(link, t, time, allLinks, viewport) {
        if (!link.control_points || link.control_points.length !== 4) {
            return { x: 0, y: 0 };
        }

        const linkIdHash = hashString(link.id);
        const reverseLink = allLinks ? findReverseLink(link, allLinks) : null;
        const biSide = getBidirectionalSide(link, reverseLink);

        let controlPoints = link.control_points;

        if (biSide !== 0) {
            controlPoints = offsetControlPoints(
                controlPoints,
                BIDIRECTIONAL_OFFSET / 2,
                biSide
            );
        }

        controlPoints = transformControlPoints(controlPoints, viewport);
        controlPoints = applySwayToControlPoints(controlPoints, link.sway_phase || 0, time, linkIdHash);

        // Cubic bezier interpolation
        const [p0, p1, p2, p3] = controlPoints;
        const mt = 1 - t;
        const mt2 = mt * mt;
        const mt3 = mt2 * mt;
        const t2 = t * t;
        const t3 = t2 * t;

        const x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0];
        const y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1];

        return { x, y };
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineLinks = {
        renderLinks: renderLinks,
        renderLink: renderLink,
        getLinkTypeColor: getLinkTypeColor,
        getPointOnLink: getPointOnLink,
        LINK_TYPE_COLORS: LINK_TYPE_COLORS
    };

})(typeof window !== 'undefined' ? window : this);
