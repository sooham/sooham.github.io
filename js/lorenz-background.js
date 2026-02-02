/**
 * Lorenz Attractor Ethereal Network Background
 * Optimized for performance with pre-rendered sprites and reduced allocations
 */

(function() {
  'use strict';

  const canvas = document.getElementById('lorenz-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d', { alpha: false });

  // Configuration
  const CONFIG = {
    sigma: 10,
    rho: 28,
    beta: 8 / 3,
    dt: 0.003,

    particleCount: 80,  // Reduced from 150 for performance
    particleMinSize: 1,
    particleMaxSize: 3.5,

    noiseAmplitude: 200,
    noiseSpeed: 0.0006,
    noiseScale: 0.5,
    lorenzInfluence: 0.3,

    connectionDistance: 70,  // Reduced from 100 for performance
    connectionDistanceSq: 4900,  // Pre-calculated squared distance (70²)
    marginPercent: 0.32,
    fadeZonePercent: 0.10,

    // Background color components (avoid string parsing)
    bgR: 250,
    bgG: 248,
    bgB: 243,
    bgA: 0.03
  };

  // FPS throttling
  const TARGET_FPS = 30;
  const FRAME_INTERVAL = 1000 / TARGET_FPS;
  let lastFrameTime = 0;

  // Pre-allocated state
  let particles;
  let opacities;  // Pre-allocated array
  let width, height, dpr;
  let animationId;
  let isVisible = true;
  let time = 0;

  // Cached calculations
  let centerX, marginWidth, fadeZone, marginInnerEdge;

  // Spatial hash using Map for better performance
  const spatialHash = new Map();
  const cellSize = CONFIG.connectionDistance;

  // Pre-rendered glow sprites (one per color)
  let glowSprites = {};
  const SPRITE_SIZE = 32;

  // Reusable array for nearby particles
  const nearbyBuffer = [];

  /**
   * Create pre-rendered glow sprite for a color
   */
  function createGlowSprite(color) {
    const size = SPRITE_SIZE;
    const spriteCanvas = document.createElement('canvas');
    spriteCanvas.width = size;
    spriteCanvas.height = size;
    const spriteCtx = spriteCanvas.getContext('2d');

    const center = size / 2;
    const gradient = spriteCtx.createRadialGradient(center, center, 0, center, center, center);

    // Cream center
    gradient.addColorStop(0, 'rgba(255, 252, 245, 0.9)');
    // Colored mid
    gradient.addColorStop(0.3, `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`);
    // Fade out
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    spriteCtx.fillStyle = gradient;
    spriteCtx.fillRect(0, 0, size, size);

    return spriteCanvas;
  }

  /**
   * Initialize glow sprites
   */
  function initSprites() {
    glowSprites = {
      grey: createGlowSprite({ r: 154, g: 149, b: 144 }),
      terracotta: createGlowSprite({ r: 196, g: 93, b: 58 })
    };
  }

  /**
   * Simple smooth noise using sine waves
   */
  function smoothNoise(x, y, t) {
    return Math.sin(x * 1.3 + t) * Math.cos(y * 0.9 + t * 0.7) * 0.5 +
           Math.sin(x * 2.1 - t * 0.5) * Math.sin(y * 1.7 + t * 0.3) * 0.3 +
           Math.cos(x * 0.8 + y * 1.1 + t * 0.9) * 0.2;
  }

  /**
   * Initialize canvas and cache calculations
   */
  function initCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);  // Cap at 2x for performance
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    ctx.scale(dpr, dpr);

    // Cache calculations
    centerX = width / 2;
    marginWidth = width * CONFIG.marginPercent;
    fadeZone = width * CONFIG.fadeZonePercent;
    marginInnerEdge = centerX - marginWidth;
  }

  /**
   * Create particle with flat structure for cache efficiency
   */
  function createParticle(side) {
    return {
      x: (Math.random() - 0.5) * 30,
      y: (Math.random() - 0.5) * 30,
      z: Math.random() * 30 + 10,
      screenX: 0,
      screenY: 0,
      noiseSeedX: Math.random() * 100,
      noiseSeedY: Math.random() * 100,
      size: CONFIG.particleMinSize + Math.random() * (CONFIG.particleMaxSize - CONFIG.particleMinSize),
      brightness: 0.4 + Math.random() * 0.6,
      colorType: Math.random() < 0.9 ? 'grey' : 'terracotta',
      side: side,
      depthScale: 1
    };
  }

  /**
   * Initialize particles and pre-allocate arrays
   */
  function initParticles() {
    const count = CONFIG.particleCount;
    particles = new Array(count);
    opacities = new Float32Array(count);  // Typed array for better performance

    const halfCount = count >> 1;  // Bitwise divide by 2

    for (let i = 0; i < halfCount; i++) {
      particles[i] = createParticle(0);
    }
    for (let i = halfCount; i < count; i++) {
      particles[i] = createParticle(1);
    }
  }

  /**
   * Update Lorenz system (inlined constants for speed)
   */
  function updateLorenz(p) {
    const dx = 10 * (p.y - p.x) * 0.003;
    const dy = (p.x * (28 - p.z) - p.y) * 0.003;
    const dz = (p.x * p.y - 2.6667 * p.z) * 0.003;

    p.x += dx;
    p.y += dy;
    p.z += dz;

    // Reset if out of bounds
    if (p.x < -50 || p.x > 50 || p.y < -50 || p.y > 50 || p.z > 80 || p.z < 0) {
      p.x = (Math.random() - 0.5) * 20;
      p.y = (Math.random() - 0.5) * 20;
      p.z = Math.random() * 20 + 15;
    }
  }

  /**
   * Project to screen with noise spreading
   */
  function projectToScreen(p) {
    const normX = (p.x + 25) * 0.02;  // / 50
    const normY = (p.y + 25) * 0.02;
    const normZ = p.z * 0.02;

    const noiseX = smoothNoise(p.noiseSeedX * 0.5, p.noiseSeedY * 0.5, time);
    const noiseY = smoothNoise(p.noiseSeedY * 0.5 + 50, p.noiseSeedX * 0.5 + 50, time * 0.8);

    const baseX = p.noiseSeedX % 1;
    const baseY = p.noiseSeedY % 1;

    const blendedX = baseX * 0.7 + normX * 0.3;
    const blendedY = baseY * 0.7 + normY * 0.3;

    const finalX = blendedX + noiseX * 0.15;
    const finalY = blendedY + noiseY * 0.2;

    if (p.side === 0) {
      p.screenX = finalX * marginWidth * 1.2;
      if (p.screenX < -20) p.screenX = -20;
      else if (p.screenX > marginWidth + width * 0.08) p.screenX = marginWidth + width * 0.08;
    } else {
      p.screenX = width - marginWidth - width * 0.08 + finalX * marginWidth * 1.2;
      if (p.screenX > width + 20) p.screenX = width + 20;
      else if (p.screenX < width - marginWidth - width * 0.08) p.screenX = width - marginWidth - width * 0.08;
    }

    const padding = height * 0.02;
    p.screenY = padding + finalY * (height - padding * 2);
    if (p.screenY < -20) p.screenY = -20;
    else if (p.screenY > height + 20) p.screenY = height + 20;

    p.depthScale = 0.7 + normZ * 0.3;
  }

  /**
   * Calculate margin opacity (inlined)
   */
  function calculateOpacity(screenX) {
    const distFromCenter = screenX > centerX ? screenX - centerX : centerX - screenX;

    if (distFromCenter < marginInnerEdge) return 0;

    const posInMargin = distFromCenter - marginInnerEdge;
    if (posInMargin < fadeZone) return posInMargin / fadeZone;

    return 1;
  }

  /**
   * Build spatial hash using Map (cleared and reused)
   */
  function buildSpatialHash() {
    spatialHash.clear();

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const cellX = (p.screenX / cellSize) | 0;  // Bitwise floor
      const cellY = (p.screenY / cellSize) | 0;
      const key = (cellX << 16) | (cellY & 0xFFFF);  // Pack into single number

      let cell = spatialHash.get(key);
      if (!cell) {
        cell = [];
        spatialHash.set(key, cell);
      }
      cell.push(i);
    }
  }

  /**
   * Get nearby particles (reuses buffer array)
   */
  function getNearbyParticles(p, buffer) {
    buffer.length = 0;
    const cellX = (p.screenX / cellSize) | 0;
    const cellY = (p.screenY / cellSize) | 0;

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const key = ((cellX + dx) << 16) | ((cellY + dy) & 0xFFFF);
        const cell = spatialHash.get(key);
        if (cell) {
          for (let i = 0; i < cell.length; i++) {
            buffer.push(cell[i]);
          }
        }
      }
    }
    return buffer;
  }

  /**
   * Draw particle using pre-rendered sprite
   */
  function drawParticle(p, opacity) {
    const size = p.size * p.depthScale * 4;
    const finalOpacity = opacity * p.brightness;

    ctx.globalAlpha = finalOpacity;
    ctx.drawImage(
      glowSprites[p.colorType],
      p.screenX - size,
      p.screenY - size,
      size * 2,
      size * 2
    );
  }

  /**
   * Main render loop (throttled to 30 FPS)
   */
  function render(currentTime) {
    if (!isVisible) {
      animationId = requestAnimationFrame(render);
      return;
    }

    // FPS throttling - skip frame if not enough time has passed
    if (currentTime - lastFrameTime < FRAME_INTERVAL) {
      animationId = requestAnimationFrame(render);
      return;
    }
    lastFrameTime = currentTime;

    time += CONFIG.noiseSpeed;

    // Clear with background color
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = CONFIG.bgA;
    ctx.fillStyle = `rgb(${CONFIG.bgR},${CONFIG.bgG},${CONFIG.bgB})`;
    ctx.fillRect(0, 0, width, height);

    // Update particles and calculate opacities
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      updateLorenz(p);
      projectToScreen(p);
      opacities[i] = calculateOpacity(p.screenX);
    }

    buildSpatialHash();

    // Set up for drawing (using source-over for better performance)
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;

    // Draw connections
    ctx.lineWidth = 0.5;
    for (let i = 0; i < particles.length; i++) {
      const p1 = particles[i];
      const opacity1 = opacities[i];

      if (opacity1 < 0.01) continue;

      const nearby = getNearbyParticles(p1, nearbyBuffer);

      for (let k = 0; k < nearby.length; k++) {
        const j = nearby[k];
        if (j <= i) continue;

        const p2 = particles[j];
        if (p1.side !== p2.side) continue;

        const opacity2 = opacities[j];
        if (opacity2 < 0.01) continue;

        const dx = p2.screenX - p1.screenX;
        const dy = p2.screenY - p1.screenY;
        const distSq = dx * dx + dy * dy;

        if (distSq < CONFIG.connectionDistanceSq) {
          const dist = Math.sqrt(distSq);
          const alpha = (1 - dist / CONFIG.connectionDistance) * 0.25 * Math.min(opacity1, opacity2);

          if (alpha > 0.01) {
            ctx.strokeStyle = `rgba(180,175,168,${alpha})`;
            ctx.beginPath();
            ctx.moveTo(p1.screenX, p1.screenY);
            ctx.lineTo(p2.screenX, p2.screenY);
            ctx.stroke();
          }
        }
      }
    }

    // Draw particles
    for (let i = 0; i < particles.length; i++) {
      if (opacities[i] > 0.01) {
        drawParticle(particles[i], opacities[i]);
      }
    }

    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;

    animationId = requestAnimationFrame(render);
  }

  function handleResize() {
    initCanvas();
    ctx.clearRect(0, 0, width, height);
  }

  function handleVisibilityChange() {
    isVisible = !document.hidden;
  }

  function init() {
    initCanvas();
    initSprites();
    initParticles();

    window.addEventListener('resize', handleResize);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
