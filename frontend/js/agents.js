/**
 * LoopEngine Agent Rendering Module
 *
 * Renders agents as breathing amoeba shapes with organic Perlin noise distortion,
 * radial gradients, and glow effects indicating load/stress.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Perlin Noise Implementation
    // =========================================================================

    // Permutation table for Perlin noise
    const permutation = [];
    const p = new Array(512);

    /**
     * Initialize Perlin noise permutation table with a seed.
     * @param {number} seed - Random seed for noise generation
     */
    function initPerlin(seed) {
        // Simple seeded random
        const random = function() {
            seed = (seed * 16807) % 2147483647;
            return (seed - 1) / 2147483646;
        };

        // Create permutation array
        for (let i = 0; i < 256; i++) {
            permutation[i] = i;
        }

        // Shuffle using Fisher-Yates
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
        }

        // Duplicate for overflow
        for (let i = 0; i < 512; i++) {
            p[i] = permutation[i & 255];
        }
    }

    /**
     * Fade function for smooth interpolation.
     * @param {number} t - Input value
     * @returns {number} Smoothed value
     */
    function fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    /**
     * Linear interpolation.
     * @param {number} t - Interpolation factor
     * @param {number} a - Start value
     * @param {number} b - End value
     * @returns {number} Interpolated value
     */
    function lerp(t, a, b) {
        return a + t * (b - a);
    }

    /**
     * Gradient function for 2D Perlin noise.
     * @param {number} hash - Hash value
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {number} Gradient value
     */
    function grad2d(hash, x, y) {
        const h = hash & 3;
        const u = h < 2 ? x : y;
        const v = h < 2 ? y : x;
        return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
    }

    /**
     * 2D Perlin noise function.
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {number} Noise value between -1 and 1
     */
    function perlin2d(x, y) {
        // Find unit square containing point
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;

        // Find relative position in square
        x -= Math.floor(x);
        y -= Math.floor(y);

        // Compute fade curves
        const u = fade(x);
        const v = fade(y);

        // Hash coordinates of square corners
        const A = p[X] + Y;
        const B = p[X + 1] + Y;

        // Interpolate
        return lerp(v,
            lerp(u, grad2d(p[A], x, y), grad2d(p[B], x - 1, y)),
            lerp(u, grad2d(p[A + 1], x, y - 1), grad2d(p[B + 1], x - 1, y - 1))
        );
    }

    // Initialize Perlin noise with default seed
    initPerlin(12345);

    // =========================================================================
    // Role Colors
    // =========================================================================

    const ROLE_COLORS = {
        owner: '#9b59b6',           // Purple
        sandwich_maker: '#e67e22',  // Orange
        cashier: '#27ae60',         // Green
        customer: '#3498db',        // Blue
        manager: '#e74c3c',         // Red
        server: '#1abc9c',          // Teal
        chef: '#f39c12',            // Yellow-orange
        default: '#4a90d9'          // Default blue
    };

    /**
     * Get color for agent role.
     * @param {string} role - Agent role
     * @returns {string} Hex color string
     */
    function getRoleColor(role) {
        return ROLE_COLORS[role] || ROLE_COLORS.default;
    }

    /**
     * Parse hex color to RGB components.
     * @param {string} hex - Hex color string
     * @returns {Object} RGB components {r, g, b}
     */
    function hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 74, g: 144, b: 217 };
    }

    // =========================================================================
    // Agent Rendering
    // =========================================================================

    /**
     * Calculate breathing radius based on phase.
     * @param {number} baseRadius - Base radius
     * @param {number} phase - Breathing phase (0 to 2π)
     * @returns {number} Animated radius
     */
    function calculateBreathingRadius(baseRadius, phase) {
        // 5% oscillation as per PRD
        return baseRadius * (1.0 + 0.05 * Math.sin(phase));
    }

    /**
     * Generate amoeba shape points using Perlin noise.
     * @param {number} cx - Center X
     * @param {number} cy - Center Y
     * @param {number} radius - Base radius
     * @param {number} numPoints - Number of points around the shape
     * @param {number} noiseScale - Scale of noise displacement
     * @param {string} agentId - Agent ID for unique noise seed
     * @param {number} time - Current time for animation
     * @returns {Array} Array of {x, y} points
     */
    function generateAmoebaPoints(cx, cy, radius, numPoints, noiseScale, agentId, time) {
        const points = [];
        // Use agent ID hash for unique noise offset
        let idHash = 0;
        for (let i = 0; i < agentId.length; i++) {
            idHash = ((idHash << 5) - idHash) + agentId.charCodeAt(i);
            idHash = idHash & idHash;
        }
        const idOffset = Math.abs(idHash % 1000);

        for (let i = 0; i < numPoints; i++) {
            const angle = (i / numPoints) * Math.PI * 2;

            // Sample Perlin noise at this angle
            // Use angle as one dimension, agent ID offset as another
            const noiseX = Math.cos(angle) * 0.5 + idOffset;
            const noiseY = Math.sin(angle) * 0.5 + time * 0.1;

            // Get noise value (range -1 to 1)
            const noise = perlin2d(noiseX, noiseY);

            // Apply noise to radius (scaled displacement)
            const displacedRadius = radius + noise * noiseScale * radius;

            // Calculate point position
            points.push({
                x: cx + Math.cos(angle) * displacedRadius,
                y: cy + Math.sin(angle) * displacedRadius
            });
        }

        return points;
    }

    /**
     * Draw smooth curve through points using Catmull-Rom spline.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} points - Array of {x, y} points
     */
    function drawSmoothShape(ctx, points) {
        if (points.length < 3) return;

        ctx.beginPath();

        // Start at first point
        ctx.moveTo(points[0].x, points[0].y);

        // Draw Catmull-Rom spline through all points
        for (let i = 0; i < points.length; i++) {
            const p0 = points[(i - 1 + points.length) % points.length];
            const p1 = points[i];
            const p2 = points[(i + 1) % points.length];
            const p3 = points[(i + 2) % points.length];

            // Calculate control points for cubic bezier
            const cp1x = p1.x + (p2.x - p0.x) / 6;
            const cp1y = p1.y + (p2.y - p0.y) / 6;
            const cp2x = p2.x - (p3.x - p1.x) / 6;
            const cp2y = p2.y - (p3.y - p1.y) / 6;

            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
        }

        ctx.closePath();
    }

    /**
     * Create radial gradient for agent fill.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} cx - Center X
     * @param {number} cy - Center Y
     * @param {number} radius - Agent radius
     * @param {string} color - Base color (hex)
     * @param {number} glowIntensity - Glow intensity (0-1)
     * @returns {CanvasGradient} Radial gradient
     */
    function createAgentGradient(ctx, cx, cy, radius, color, glowIntensity) {
        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);

        const rgb = hexToRgb(color);

        // Brighten center based on glow intensity
        const centerBrightness = Math.min(255, rgb.r + 80 + glowIntensity * 50);
        const centerG = Math.min(255, rgb.g + 80 + glowIntensity * 50);
        const centerB = Math.min(255, rgb.b + 80 + glowIntensity * 50);

        // Glow intensity affects inner brightness
        const innerAlpha = 0.9 + glowIntensity * 0.1;
        const outerAlpha = 0.6 + glowIntensity * 0.2;

        gradient.addColorStop(0, `rgba(${centerBrightness}, ${centerG}, ${centerB}, ${innerAlpha})`);
        gradient.addColorStop(0.5, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${(innerAlpha + outerAlpha) / 2})`);
        gradient.addColorStop(1, `rgba(${rgb.r * 0.7}, ${rgb.g * 0.7}, ${rgb.b * 0.7}, ${outerAlpha})`);

        return gradient;
    }

    /**
     * Draw outer glow effect for agent.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} cx - Center X
     * @param {number} cy - Center Y
     * @param {number} radius - Agent radius
     * @param {string} color - Base color (hex)
     * @param {number} glowIntensity - Glow intensity (0-1)
     */
    function drawGlow(ctx, cx, cy, radius, color, glowIntensity) {
        if (glowIntensity < 0.01) return;

        const rgb = hexToRgb(color);
        const glowRadius = radius + 10 + glowIntensity * 15;

        ctx.save();

        // Create glow gradient
        const glowGradient = ctx.createRadialGradient(cx, cy, radius * 0.8, cx, cy, glowRadius);
        glowGradient.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.3 * glowIntensity})`);
        glowGradient.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);

        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    /**
     * Render a single agent on the canvas.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} agent - AgentVisual object from frame
     * @param {number} time - Current animation time
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     */
    function renderAgent(ctx, agent, time, viewport) {
        const scale = viewport ? viewport.scale : 1;
        const offsetX = viewport ? viewport.offsetX : 0;
        const offsetY = viewport ? viewport.offsetY : 0;

        // Get hover state from interaction module
        let hoverState = { glowBoost: 0, scaleBoost: 1.0, isHovered: false };
        if (typeof LoopEngineInteraction !== 'undefined') {
            hoverState = LoopEngineInteraction.getAgentHoverState(agent);
        }

        // Transform coordinates to screen space
        const screenX = agent.x * scale + offsetX;
        const screenY = agent.y * scale + offsetY;

        // Calculate breathing radius with hover scale boost
        const baseRadius = agent.radius || 20;
        const breathingRadius = calculateBreathingRadius(baseRadius, agent.breathing_phase || 0);
        const scaledRadius = breathingRadius * scale * hoverState.scaleBoost;

        // Skip if too small to see
        if (scaledRadius < 1) return;

        // Get role color
        const color = agent.color || getRoleColor(agent.role);

        // Calculate glow with hover boost
        const effectiveGlow = Math.min(1.0, (agent.glow_intensity || 0) + hoverState.glowBoost);

        // Draw outer glow first (underneath)
        drawGlow(ctx, screenX, screenY, scaledRadius, color, effectiveGlow);

        // Generate amoeba shape points
        const numPoints = 24;  // Enough points for smooth shape
        const noiseScale = 0.15;  // 15% max displacement from noise
        const points = generateAmoebaPoints(
            screenX, screenY,
            scaledRadius,
            numPoints,
            noiseScale,
            agent.id,
            time
        );

        // Create gradient fill with hover-boosted glow
        const gradient = createAgentGradient(ctx, screenX, screenY, scaledRadius, color, effectiveGlow);

        // Draw the amoeba shape
        ctx.save();

        ctx.fillStyle = gradient;
        drawSmoothShape(ctx, points);
        ctx.fill();

        // Draw subtle membrane stroke
        ctx.strokeStyle = `rgba(255, 255, 255, 0.2)`;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.restore();

        // Draw agent name label (only if zoomed in enough)
        if (scale >= 0.5) {
            drawAgentLabel(ctx, screenX, screenY + scaledRadius + 12, agent.name, scale);
        }
    }

    /**
     * Draw agent name label below the agent.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - Label center X
     * @param {number} y - Label top Y
     * @param {string} name - Agent name
     * @param {number} scale - Current zoom scale
     */
    function drawAgentLabel(ctx, x, y, name, scale) {
        ctx.save();

        // Scale font size with zoom but clamp to reasonable range
        const fontSize = Math.max(8, Math.min(14, 12 * scale));

        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // Draw text shadow for readability
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillText(name, x + 1, y + 1);

        // Draw text
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.fillText(name, x, y);

        ctx.restore();
    }

    /**
     * Render all agents in a frame.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} agents - Array of AgentVisual objects
     * @param {number} time - Current animation time
     * @param {Object} viewport - Viewport transform {scale, offsetX, offsetY}
     */
    function renderAgents(ctx, agents, time, viewport) {
        if (!agents || agents.length === 0) return;

        // Sort agents by Y position for basic depth ordering
        const sortedAgents = [...agents].sort((a, b) => a.y - b.y);

        for (const agent of sortedAgents) {
            renderAgent(ctx, agent, time, viewport);
        }
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineAgents = {
        renderAgents: renderAgents,
        renderAgent: renderAgent,
        getRoleColor: getRoleColor,
        calculateBreathingRadius: calculateBreathingRadius,
        initPerlin: initPerlin  // Allow re-seeding if needed
    };

})(typeof window !== 'undefined' ? window : this);
