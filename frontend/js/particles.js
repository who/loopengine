/**
 * LoopEngine Particle Rendering Module
 *
 * Renders particles as colored dots with fading trails along link bezier paths.
 * Particle types are color-coded per PRD section 7.4.
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Particle Type Colors (per PRD section 7.4)
    // =========================================================================

    const PARTICLE_TYPE_COLORS = {
        order_ticket: '#f0a500',      // Warm amber
        customer_order: '#f0a500',    // Warm amber (alias)
        finished_sandwich: '#27ae60', // Green
        sandwich: '#27ae60',          // Green (alias)
        directive: '#3498db',         // Cool blue
        status_report: '#bdc3c7',     // Light gray
        supply_order: '#e67e22',      // Orange
        revenue_report: '#f1c40f',    // Gold
        money: '#f1c40f',             // Gold (alias)
        ingredient_delivery: '#9b59b6', // Purple
        served_customer: '#2ecc71',   // Bright green
        stockout_alert: '#e74c3c',    // Red
        default: '#ffffff'            // White fallback
    };

    // =========================================================================
    // Particle Size Configuration (radius in pixels by type)
    // =========================================================================

    const PARTICLE_TYPE_SIZES = {
        order_ticket: 3,
        customer_order: 3,
        finished_sandwich: 4,
        sandwich: 4,
        directive: 3,
        status_report: 2,
        supply_order: 4,
        revenue_report: 3,
        money: 3,
        ingredient_delivery: 5,
        served_customer: 3,
        stockout_alert: 3,
        default: 3
    };

    // Trail configuration
    const TRAIL_LENGTH = 5;           // Number of trail positions
    const TRAIL_OPACITY_DECAY = 0.7;  // Each trail position is this much dimmer

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /**
     * Get color for a particle type.
     * @param {string} particleType - The particle type
     * @returns {string} Hex color string
     */
    function getParticleColor(particleType) {
        return PARTICLE_TYPE_COLORS[particleType] || PARTICLE_TYPE_COLORS.default;
    }

    /**
     * Get size for a particle type.
     * @param {string} particleType - The particle type
     * @returns {number} Radius in pixels
     */
    function getParticleSize(particleType) {
        return PARTICLE_TYPE_SIZES[particleType] || PARTICLE_TYPE_SIZES.default;
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
        } : { r: 255, g: 255, b: 255 };
    }

    /**
     * Find a link by ID in the links array.
     * @param {string} linkId - Link ID to find
     * @param {Array} links - Array of link objects
     * @returns {Object|null} Link object or null
     */
    function findLinkById(linkId, links) {
        return links.find(link => link.id === linkId) || null;
    }

    // =========================================================================
    // Position Calculation
    // =========================================================================

    /**
     * Calculate particle position along a link's bezier path.
     * Uses the LoopEngineLinks module if available for consistent positioning.
     * @param {Object} particle - Particle object with link_id and progress
     * @param {Array} links - All links in the frame
     * @param {number} time - Animation time
     * @param {Object} viewport - Viewport transform
     * @returns {Object|null} Position {x, y} or null if link not found
     */
    function calculateParticlePosition(particle, links, time, viewport) {
        const link = findLinkById(particle.link_id, links);
        if (!link) {
            return null;
        }

        // Use LoopEngineLinks for consistent bezier interpolation if available
        if (typeof LoopEngineLinks !== 'undefined' && LoopEngineLinks.getPointOnLink) {
            return LoopEngineLinks.getPointOnLink(link, particle.progress, time, links, viewport);
        }

        // Fallback: simple linear interpolation if links module not available
        if (!link.control_points || link.control_points.length !== 4) {
            return null;
        }

        const scale = viewport ? viewport.scale : 1;
        const offsetX = viewport ? viewport.offsetX : 0;
        const offsetY = viewport ? viewport.offsetY : 0;

        // Cubic bezier interpolation
        const t = particle.progress;
        const [p0, p1, p2, p3] = link.control_points;
        const mt = 1 - t;
        const mt2 = mt * mt;
        const mt3 = mt2 * mt;
        const t2 = t * t;
        const t3 = t2 * t;

        const x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0];
        const y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1];

        return {
            x: x * scale + offsetX,
            y: y * scale + offsetY
        };
    }

    // =========================================================================
    // Rendering Functions
    // =========================================================================

    /**
     * Draw a particle trail.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} trail - Array of {x, y} positions (most recent first)
     * @param {string} color - Particle color (hex)
     * @param {number} baseSize - Base particle size
     * @param {number} scale - Viewport scale
     */
    function drawTrail(ctx, trail, color, baseSize, scale) {
        if (!trail || trail.length < 2) return;

        const rgb = hexToRgb(color);
        let opacity = 0.6;

        // Draw trail from oldest to newest (so newer overlaps older)
        for (let i = trail.length - 1; i >= 1; i--) {
            const pos = trail[i];
            const size = Math.max(1, baseSize * scale * (1 - (i / trail.length) * 0.5));

            ctx.beginPath();
            ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
            ctx.fill();

            opacity *= TRAIL_OPACITY_DECAY;
        }
    }

    /**
     * Draw a single particle with glow effect.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} x - X position
     * @param {number} y - Y position
     * @param {string} color - Particle color (hex)
     * @param {number} size - Particle radius
     * @param {number} scale - Viewport scale
     * @param {number} opacity - Base opacity (0-1)
     */
    function drawParticle(ctx, x, y, color, size, scale, opacity) {
        const scaledSize = Math.max(1.5, size * Math.sqrt(scale));
        const rgb = hexToRgb(color);

        ctx.save();

        // Outer glow
        const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, scaledSize * 2);
        glowGradient.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity * 0.4})`);
        glowGradient.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(x, y, scaledSize * 2, 0, Math.PI * 2);
        ctx.fill();

        // Main particle body
        const bodyGradient = ctx.createRadialGradient(x, y, 0, x, y, scaledSize);
        bodyGradient.addColorStop(0, `rgba(255, 255, 255, ${opacity})`);
        bodyGradient.addColorStop(0.3, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`);
        bodyGradient.addColorStop(1, `rgba(${rgb.r * 0.7 | 0}, ${rgb.g * 0.7 | 0}, ${rgb.b * 0.7 | 0}, ${opacity * 0.8})`);
        ctx.fillStyle = bodyGradient;
        ctx.beginPath();
        ctx.arc(x, y, scaledSize, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    /**
     * Render a single particle with its trail.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} particle - ParticleVisual object from frame
     * @param {Array} links - All links in the frame
     * @param {number} time - Animation time
     * @param {Object} viewport - Viewport transform
     */
    function renderParticle(ctx, particle, links, time, viewport) {
        // Skip dead particles
        if (particle.alive === false) return;

        const scale = viewport ? viewport.scale : 1;
        const color = particle.color || getParticleColor(particle.particle_type);
        const size = particle.size || getParticleSize(particle.particle_type);
        const opacity = particle.opacity !== undefined ? particle.opacity : 1.0;

        // Get current position
        let pos;
        if (particle.x !== undefined && particle.y !== undefined) {
            // Use pre-computed position from frame if available
            const offsetX = viewport ? viewport.offsetX : 0;
            const offsetY = viewport ? viewport.offsetY : 0;
            pos = {
                x: particle.x * scale + offsetX,
                y: particle.y * scale + offsetY
            };
        } else {
            // Calculate position from link and progress
            pos = calculateParticlePosition(particle, links, time, viewport);
        }

        if (!pos) return;

        // Draw trail first (behind particle)
        if (particle.trail && particle.trail.length > 0) {
            // Transform trail positions to screen coordinates
            const offsetX = viewport ? viewport.offsetX : 0;
            const offsetY = viewport ? viewport.offsetY : 0;
            const screenTrail = particle.trail.map(p => ({
                x: p[0] * scale + offsetX,
                y: p[1] * scale + offsetY
            }));
            // Add current position at the front
            screenTrail.unshift(pos);
            drawTrail(ctx, screenTrail, color, size, scale);
        }

        // Draw the main particle
        drawParticle(ctx, pos.x, pos.y, color, size, scale, opacity);
    }

    /**
     * Render all particles in a frame.
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} particles - Array of ParticleVisual objects
     * @param {Array} links - Array of LinkVisual objects
     * @param {number} time - Animation time
     * @param {Object} viewport - Viewport transform
     */
    function renderParticles(ctx, particles, links, time, viewport) {
        if (!particles || particles.length === 0) return;

        // Sort particles by size (draw larger ones first, smaller on top)
        const sortedParticles = [...particles].sort((a, b) => {
            const sizeA = a.size || getParticleSize(a.particle_type);
            const sizeB = b.size || getParticleSize(b.particle_type);
            return sizeB - sizeA;
        });

        for (const particle of sortedParticles) {
            renderParticle(ctx, particle, links, time, viewport);
        }
    }

    // =========================================================================
    // Module Export
    // =========================================================================

    global.LoopEngineParticles = {
        renderParticles: renderParticles,
        renderParticle: renderParticle,
        getParticleColor: getParticleColor,
        getParticleSize: getParticleSize,
        PARTICLE_TYPE_COLORS: PARTICLE_TYPE_COLORS,
        PARTICLE_TYPE_SIZES: PARTICLE_TYPE_SIZES
    };

})(typeof window !== 'undefined' ? window : this);
