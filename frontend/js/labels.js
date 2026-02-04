/**
 * LoopEngine Label Region Rendering Module
 *
 * Renders label regions as soft translucent clouds around agents sharing labels.
 * Uses convex hull expansion, cardinal spline smoothing, and breathing animation.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Configuration
    // =========================================================================

    const HULL_EXPANSION = 25;         // Pixels to expand hull outward
    const BREATHING_AMPLITUDE = 0.03;  // 3% breathing oscillation
    const FILL_OPACITY_MIN = 0.08;
    const FILL_OPACITY_MAX = 0.15;
    const SMOOTHING_TENSION = 0.4;     // Cardinal spline tension (0 = sharp, 1 = very smooth)
    const SMOOTHING_POINTS = 6;        // Interpolated points between each hull vertex

    // Default colors for label regions
    const LABEL_COLORS = {
        kitchen: '#ff9f43',
        counter: '#00d2d3',
        shop: '#5f27cd',
        default: '#88ccff'
    };

    // =========================================================================
    // Hull Manipulation Functions
    // =========================================================================

    /**
     * Expand convex hull outward from its centroid.
     * @param {Array} points - Array of [x, y] points
     * @param {number} expansion - Distance to expand
     * @returns {Array} Expanded hull points
     */
    function expandHull(points, expansion) {
        if (points.length < 3) return points;

        // Calculate centroid
        let cx = 0, cy = 0;
        for (const [x, y] of points) {
            cx += x;
            cy += y;
        }
        cx /= points.length;
        cy /= points.length;

        // Expand each point outward from centroid
        const expanded = [];
        for (const [x, y] of points) {
            const dx = x - cx;
            const dy = y - cy;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 0.001) {
                expanded.push([x + expansion, y]);
            } else {
                const scale = (dist + expansion) / dist;
                expanded.push([
                    cx + dx * scale,
                    cy + dy * scale
                ]);
            }
        }

        return expanded;
    }

    /**
     * Apply breathing animation to hull points.
     * @param {Array} points - Array of [x, y] points
     * @param {number} breathingPhase - Current breathing phase (0 to 2π)
     * @returns {Array} Animated hull points
     */
    function applyBreathing(points, breathingPhase) {
        if (points.length < 3) return points;

        // Calculate centroid
        let cx = 0, cy = 0;
        for (const [x, y] of points) {
            cx += x;
            cy += y;
        }
        cx /= points.length;
        cy /= points.length;

        // Apply breathing scale
        const scale = 1.0 + BREATHING_AMPLITUDE * Math.sin(breathingPhase);

        return points.map(([x, y]) => {
            const dx = x - cx;
            const dy = y - cy;
            return [cx + dx * scale, cy + dy * scale];
        });
    }

    /**
     * Smooth hull using cardinal spline interpolation.
     * Creates a rounded shape from angular hull points.
     * @param {Array} points - Array of [x, y] points
     * @param {number} tension - Spline tension (0-1)
     * @param {number} numPoints - Points to interpolate between vertices
     * @returns {Array} Smoothed curve points
     */
    function smoothHull(points, tension, numPoints) {
        if (points.length < 3) return points;

        const smoothed = [];
        const n = points.length;

        // Cardinal spline through closed loop
        for (let i = 0; i < n; i++) {
            const p0 = points[(i - 1 + n) % n];
            const p1 = points[i];
            const p2 = points[(i + 1) % n];
            const p3 = points[(i + 2) % n];

            // Interpolate between p1 and p2
            for (let j = 0; j < numPoints; j++) {
                const t = j / numPoints;
                const point = cardinalSplinePoint(p0, p1, p2, p3, t, tension);
                smoothed.push(point);
            }
        }

        return smoothed;
    }

    /**
     * Calculate a point on a cardinal spline.
     * @param {Array} p0 - Control point before segment start
     * @param {Array} p1 - Segment start
     * @param {Array} p2 - Segment end
     * @param {Array} p3 - Control point after segment end
     * @param {number} t - Parameter 0-1
     * @param {number} tension - Spline tension 0-1
     * @returns {Array} Interpolated [x, y] point
     */
    function cardinalSplinePoint(p0, p1, p2, p3, t, tension) {
        const t2 = t * t;
        const t3 = t2 * t;

        const s = (1 - tension) / 2;

        // Cardinal spline basis functions
        const h1 = 2 * t3 - 3 * t2 + 1;
        const h2 = -2 * t3 + 3 * t2;
        const h3 = t3 - 2 * t2 + t;
        const h4 = t3 - t2;

        // Tangents scaled by tension
        const m1x = s * (p2[0] - p0[0]);
        const m1y = s * (p2[1] - p0[1]);
        const m2x = s * (p3[0] - p1[0]);
        const m2y = s * (p3[1] - p1[1]);

        const x = h1 * p1[0] + h2 * p2[0] + h3 * m1x + h4 * m2x;
        const y = h1 * p1[1] + h2 * p2[1] + h3 * m1y + h4 * m2y;

        return [x, y];
    }

    // =========================================================================
    // Color Utilities
    // =========================================================================

    /**
     * Parse color string to RGBA components.
     * @param {string} color - Color string (hex with optional alpha)
     * @returns {Object} RGBA components {r, g, b, a}
     */
    function parseColor(color) {
        // Handle rgba format
        const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (rgbaMatch) {
            return {
                r: parseInt(rgbaMatch[1], 10),
                g: parseInt(rgbaMatch[2], 10),
                b: parseInt(rgbaMatch[3], 10),
                a: rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1.0
            };
        }

        // Handle hex format (with optional alpha)
        let hex = color.replace('#', '');
        let a = 1.0;

        if (hex.length === 8) {
            a = parseInt(hex.substring(6, 8), 16) / 255;
            hex = hex.substring(0, 6);
        }

        if (hex.length === 3) {
            hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
        }

        const result = /^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        if (result) {
            return {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16),
                a: a
            };
        }

        return { r: 136, g: 204, b: 255, a: 0.2 };
    }

    /**
     * Get color for a label region.
     * @param {string} name - Label name
     * @param {string} fillColor - Backend-provided color (optional)
     * @returns {Object} RGBA components
     */
    function getLabelColor(name, fillColor) {
        if (fillColor) {
            return parseColor(fillColor);
        }

        const baseName = name.toLowerCase();
        for (const [key, value] of Object.entries(LABEL_COLORS)) {
            if (baseName.includes(key)) {
                return parseColor(value);
            }
        }

        return parseColor(LABEL_COLORS.default);
    }

    // =========================================================================
    // Rendering Functions
    // =========================================================================

    /**
     * Draw a smoothed filled shape.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} points - Array of [x, y] points
     */
    function drawFilledShape(ctx, points) {
        if (points.length < 3) return;

        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);

        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
        }

        ctx.closePath();
        ctx.fill();
    }

    /**
     * Transform hull points to screen coordinates.
     * @param {Array} points - Hull points in world coordinates
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     * @returns {Array} Screen-space points
     */
    function transformPoints(points, viewport) {
        const scale = viewport ? viewport.scale : 1;
        const offsetX = viewport ? viewport.offsetX : 0;
        const offsetY = viewport ? viewport.offsetY : 0;

        return points.map(([x, y]) => [
            x * scale + offsetX,
            y * scale + offsetY
        ]);
    }

    /**
     * Render a single label region on the canvas.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} region - LabelRegionVisual object from frame
     * @param {number} time - Current animation time
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     */
    function renderLabelRegion(ctx, region, time, viewport) {
        // Skip if no hull points
        if (!region.hull_points || region.hull_points.length < 3) {
            return;
        }

        const scale = viewport ? viewport.scale : 1;

        // Convert backend format (array of [x, y]) to our format
        let points = region.hull_points.map(p => [p[0], p[1]]);

        // Transform to screen coordinates first
        points = transformPoints(points, viewport);

        // Expand hull in screen space (scaled expansion)
        points = expandHull(points, HULL_EXPANSION * Math.sqrt(scale));

        // Apply breathing animation
        const breathingPhase = region.breathing_phase || 0;
        points = applyBreathing(points, breathingPhase);

        // Smooth the hull into rounded shape
        const smoothed = smoothHull(points, SMOOTHING_TENSION, SMOOTHING_POINTS);

        // Get color with proper opacity
        const color = getLabelColor(region.name, region.fill_color);

        // Calculate opacity oscillating gently with breathing
        const baseOpacity = (FILL_OPACITY_MIN + FILL_OPACITY_MAX) / 2;
        const opacityVariation = (FILL_OPACITY_MAX - FILL_OPACITY_MIN) / 2;
        const opacity = baseOpacity + opacityVariation * Math.sin(breathingPhase);

        ctx.save();

        // Set blend mode for additive blending where regions overlap
        ctx.globalCompositeOperation = 'lighter';

        // Fill with translucent color
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
        drawFilledShape(ctx, smoothed);

        ctx.restore();
    }

    /**
     * Render all label regions in a frame.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} labelRegions - Array of LabelRegionVisual objects
     * @param {number} time - Current animation time
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     */
    function renderLabelRegions(ctx, labelRegions, time, viewport) {
        if (!labelRegions || labelRegions.length === 0) return;

        // Sort by name for consistent layering
        const sortedRegions = [...labelRegions].sort((a, b) =>
            a.name.localeCompare(b.name)
        );

        for (const region of sortedRegions) {
            renderLabelRegion(ctx, region, time, viewport);
        }
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineLabels = {
        renderLabelRegions: renderLabelRegions,
        renderLabelRegion: renderLabelRegion,
        expandHull: expandHull,
        smoothHull: smoothHull,
        applyBreathing: applyBreathing,
        LABEL_COLORS: LABEL_COLORS
    };

})(typeof window !== 'undefined' ? window : this);
