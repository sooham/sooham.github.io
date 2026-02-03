/**
 * Cache Thrashing Visualizer
 * Demonstrates cache behavior during naive matrix transpose
 *
 * Models write-allocate policy: on write miss, cache line is loaded first
 * Uses LRU (Least Recently Used) replacement policy
 *
 * Basquiat-inspired styling with configurable parameters
 */

(function() {
  'use strict';

  // ===========================================
  // CONFIGURABLE PARAMETERS (adjust these)
  // ===========================================
  const CONFIG = {
    // For animation: 1 unit = 1 double (8 bytes)
    n: 8,                    // Matrix dimension (n x n)
    cacheLineSize: 4,        // Units per cache line (e.g., 4 doubles = 32 bytes)
    l1CacheSize: 32,         // Total L1 cache units
    l1ReadTime: 1,           // Seconds for L1 read
    l1WriteTime: 1,          // Seconds for L1 write
    ramReadTime: 10,         // Seconds for RAM read
    ramWriteTime: 10,        // Seconds for RAM write (after line is in cache)
  };

  // Color palette (Basquiat-inspired)
  const COLORS = {
    bgCream: '#faf8f3',
    bgPaper: '#f7f4ed',
    textInk: '#2c2824',
    textPencil: '#3d3832',
    textLight: '#6b6359',
    terracotta: '#c45d3a',
    sage: '#6b8f71',
    mustard: '#d4a84b',
    evict: '#8b4570',
  };

  // Derived values
  const MATRIX_SIZE = CONFIG.n * CONFIG.n;
  const CACHE_LINES_IN_L1 = Math.floor(CONFIG.l1CacheSize / CONFIG.cacheLineSize);

  // State
  let state = {
    playing: false,  // true = auto-advancing, false = paused/stopped
    finished: false,
    i: 0,
    j: 0,
    phase: 'read', // 'read', 'write-fetch', 'write'
    cacheHits: 0,
    cacheMisses: 0,
    evictions: 0,
    totalTime: 0,
    l1Cache: [], // Array of { source: 'src'|'dst', lineStart: number, data: number[] }
    srcCompleted: new Set(),
    dstCompleted: new Set(),
    animationId: null,
  };

  let els = {};

  // ===========================================
  // CACHE LOGIC (LRU with write-allocate)
  // ===========================================

  function getCacheLineStart(index) {
    return Math.floor(index / CONFIG.cacheLineSize) * CONFIG.cacheLineSize;
  }

  // Find a cache line by source and lineStart
  // Returns { index, entry } or null if not found
  function findInCache(source, lineStart) {
    for (let i = 0; i < state.l1Cache.length; i++) {
      if (state.l1Cache[i].source === source && state.l1Cache[i].lineStart === lineStart) {
        return { index: i, entry: state.l1Cache[i] };
      }
    }
    return null;
  }

  // Access cache line (for both read and write)
  // Returns { hit: boolean, evicted: { source, lineStart } | null }
  function accessCache(source, lineStart) {
    const found = findInCache(source, lineStart);

    if (found) {
      // Cache hit - move to end (most recently used)
      state.l1Cache.splice(found.index, 1);
      state.l1Cache.push(found.entry);
      return { hit: true, evicted: null };
    }

    // Cache miss - need to load
    let evicted = null;

    if (state.l1Cache.length >= CACHE_LINES_IN_L1) {
      // Evict LRU (first element)
      const evictedEntry = state.l1Cache.shift();
      evicted = { source: evictedEntry.source, lineStart: evictedEntry.lineStart };
      state.evictions++;
    }

    // Add new cache line at end (most recently used)
    state.l1Cache.push({
      source: source,
      lineStart: lineStart,
      data: Array.from({ length: CONFIG.cacheLineSize }, (_, i) => lineStart + i)
    });

    return { hit: false, evicted: evicted };
  }

  // Check if a line is in cache (without modifying LRU order)
  function isInCache(source, lineStart) {
    return findInCache(source, lineStart) !== null;
  }

  // ===========================================
  // HTML GENERATION
  // ===========================================

  function createHTML(container) {
    container.innerHTML = `
      <svg class="cache-crown" width="60" height="40" viewBox="0 0 60 40">
        <path d="M5 35 L15 10 L20 25 L30 5 L40 25 L45 10 L55 35" stroke="${COLORS.terracotta}" stroke-width="3" fill="none"/>
        <circle cx="15" cy="8" r="3" fill="${COLORS.mustard}"/>
        <circle cx="30" cy="3" r="3" fill="${COLORS.mustard}"/>
        <circle cx="45" cy="8" r="3" fill="${COLORS.mustard}"/>
      </svg>
      <h3 class="cache-title">CACHE THRASHING VISUALIZER</h3>
      <p class="cache-subtitle">Watch how naive transpose destroys cache efficiency. </p>
      <div class="cache-controls">
        <button id="cache-play-btn" class="cache-btn cache-btn-play">PLAY</button>
        <button id="cache-step-btn" class="cache-btn cache-btn-step" disabled>STEP</button>
        <button id="cache-reset-btn" class="cache-btn cache-btn-reset">RESET</button>
      </div>
      <div class="cache-stats">
        <div class="cache-stat">
          <div class="cache-stat-value cache-stat-hits" id="cache-hits">0</div>
          <div class="cache-stat-label">L1 HITS</div>
        </div>
        <div class="cache-stat">
          <div class="cache-stat-value cache-stat-misses" id="cache-misses">0</div>
          <div class="cache-stat-label">L1 MISSES</div>
        </div>
        <div class="cache-stat">
          <div class="cache-stat-value cache-stat-evictions" id="cache-evictions">0</div>
          <div class="cache-stat-label">EVICTIONS</div>
        </div>
        <div class="cache-stat">
          <div class="cache-stat-value cache-stat-time" id="cache-time">0s</div>
          <div class="cache-stat-label">TOTAL TIME</div>
        </div>
        <div class="cache-stat">
          <div class="cache-stat-value cache-stat-op" id="cache-op">—</div>
          <div class="cache-stat-label">OPERATION</div>
        </div>
      </div>
      <div class="cache-current-op" id="cache-current-op">
        <span class="cache-comment">// Waiting to start...</span>
      </div>
      <div class="cache-visual">
        <div class="cache-section">
          <div class="cache-section-title"><span class="cache-dot cache-dot-sage">●</span> L1 CACHE <span class="cache-section-info">(${CONFIG.l1CacheSize} units / ${CACHE_LINES_IN_L1} lines, LRU replacement)</span></div>
          <div class="cache-l1" id="cache-l1"></div>
        </div>
        <div class="cache-matrices">
          <div class="cache-section">
            <div class="cache-section-title"><span class="cache-dot cache-dot-terracotta">●</span> SRC MATRIX <span class="cache-section-info">(RAM - row major read)</span></div>
            <div class="cache-matrix" id="cache-src"></div>
          </div>
          <div class="cache-section">
            <div class="cache-section-title"><span class="cache-dot cache-dot-mustard">●</span> DST MATRIX <span class="cache-section-info">(RAM - column major write)</span></div>
            <div class="cache-matrix" id="cache-dst"></div>
          </div>
        </div>
      </div>
      <div class="cache-legend">
        <div class="cache-legend-item"><span class="cache-legend-box cache-legend-reading"></span> Reading</div>
        <div class="cache-legend-item"><span class="cache-legend-box cache-legend-loading"></span> Loading (write-allocate)</div>
        <div class="cache-legend-item"><span class="cache-legend-box cache-legend-writing"></span> Writing</div>
        <div class="cache-legend-item"><span class="cache-legend-box cache-legend-cached"></span> In Cache</div>
        <div class="cache-legend-item"><span class="cache-legend-box cache-legend-done"></span> Completed</div>
      </div>
      <div class="cache-signature">CACHE IS KING ©</div>
    `;

    injectStyles();
  }

  function injectStyles() {
    if (document.getElementById('cache-animation-styles')) return;

    const style = document.createElement('style');
    style.id = 'cache-animation-styles';
    style.textContent = `
      #cache-animation-container {
        margin: 2rem 0;
        padding: 1.5rem;
        background: ${COLORS.bgCream};
        border: 4px solid ${COLORS.textInk};
        position: relative;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }
      .cache-crown {
        position: absolute;
        top: -20px;
        left: 20px;
      }
      .cache-title {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.8rem;
        letter-spacing: 3px;
        color: ${COLORS.textInk};
        text-transform: uppercase;
        margin: 0 0 1rem 0;
        border-bottom: 3px solid ${COLORS.textInk};
        padding-bottom: 0.5rem;
      }
      .cache-subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: ${COLORS.textLight};
        margin-bottom: 1rem;
      }
      .cache-controls {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
        align-items: center;
      }
      .cache-btn {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.2rem;
        letter-spacing: 2px;
        padding: 0.5rem 1.5rem;
        border: 3px solid ${COLORS.textInk};
        cursor: pointer;
        text-transform: uppercase;
        transition: all 0.2s;
      }
      .cache-btn:hover:not(:disabled) { transform: translateY(-2px); }
      .cache-btn:disabled { opacity: 0.4; cursor: not-allowed; }
      .cache-btn-play { background: ${COLORS.sage}; color: ${COLORS.bgCream}; }
      .cache-btn-play.playing { background: ${COLORS.terracotta}; }
      .cache-btn-step { background: ${COLORS.mustard}; color: ${COLORS.textInk}; }
      .cache-btn-reset { background: ${COLORS.textPencil}; color: ${COLORS.bgCream}; }
      .cache-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: ${COLORS.bgPaper};
        border: 2px solid ${COLORS.textInk};
      }
      .cache-stat { text-align: center; }
      .cache-stat-value {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.4rem;
      }
      .cache-stat-hits { color: ${COLORS.sage}; }
      .cache-stat-misses { color: ${COLORS.terracotta}; }
      .cache-stat-evictions { color: ${COLORS.evict}; }
      .cache-stat-time { color: ${COLORS.textInk}; }
      .cache-stat-op { color: ${COLORS.mustard}; }
      .cache-stat-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: ${COLORS.textLight};
      }
      .cache-current-op {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: ${COLORS.textInk};
        color: ${COLORS.bgCream};
        border-left: 4px solid ${COLORS.terracotta};
        white-space: pre-wrap;
        line-height: 1.6;
      }
      .cache-comment { color: ${COLORS.mustard}; }
      .cache-visual {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
      }
      .cache-matrices {
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
      }
      .cache-section-title {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.2rem;
        letter-spacing: 2px;
        color: ${COLORS.textInk};
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      .cache-section-info {
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: ${COLORS.textLight};
      }
      .cache-dot-sage { color: ${COLORS.sage}; }
      .cache-dot-terracotta { color: ${COLORS.terracotta}; }
      .cache-dot-mustard { color: ${COLORS.mustard}; }
      .cache-l1 {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        padding: 0.5rem;
        background: ${COLORS.bgPaper};
        border: 3px solid ${COLORS.sage};
        min-height: 60px;
      }
      .cache-matrix {
        display: grid;
        gap: 1px;
        padding: 0.5rem;
        background: ${COLORS.bgPaper};
        grid-template-columns: repeat(${CONFIG.n}, 40px);
        width: fit-content;
      }
      #cache-src { border: 3px solid ${COLORS.terracotta}; }
      #cache-dst { border: 3px solid ${COLORS.mustard}; }
      .cache-cell {
        width: 40px;
        height: 40px;
        background: ${COLORS.bgCream};
        border: 2px solid ${COLORS.textInk};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: ${COLORS.textPencil};
        transition: all 0.15s;
        box-sizing: border-box;
      }
      .cache-cell-highlight { transform: scale(1.1); z-index: 10; }
      .cache-cell-reading { background: ${COLORS.terracotta} !important; color: ${COLORS.bgCream} !important; }
      .cache-cell-loading { background: ${COLORS.evict} !important; color: ${COLORS.bgCream} !important; }
      .cache-cell-writing { background: ${COLORS.mustard} !important; color: ${COLORS.textInk} !important; }
      .cache-cell-cached { background: rgba(107, 143, 113, 0.3); }
      .cache-cell-done { background: ${COLORS.textPencil} !important; color: ${COLORS.bgCream} !important; }
      .cache-line-cell {
        width: 28px;
        height: 28px;
        background: ${COLORS.sage};
        border: 1px solid ${COLORS.textInk};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: ${COLORS.bgCream};
      }
      .cache-line-group {
        display: flex;
        flex-wrap: nowrap;
        gap: 1px;
        padding: 4px;
        border: 2px solid;
        margin: 2px;
        position: relative;
      }
      .cache-line-group::before {
        content: attr(data-label);
        position: absolute;
        top: -10px;
        left: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        background: ${COLORS.bgPaper};
        padding: 0 2px;
      }
      .cache-line-src { background: rgba(196, 93, 58, 0.2); border-color: ${COLORS.terracotta}; }
      .cache-line-dst { background: rgba(212, 168, 75, 0.2); border-color: ${COLORS.mustard}; }
      .cache-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 2px dashed ${COLORS.textInk};
      }
      .cache-legend-item {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
      }
      .cache-legend-box {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid ${COLORS.textInk};
      }
      .cache-legend-reading { background: ${COLORS.terracotta}; }
      .cache-legend-loading { background: ${COLORS.evict}; }
      .cache-legend-writing { background: ${COLORS.mustard}; }
      .cache-legend-cached { background: ${COLORS.sage}; }
      .cache-legend-done { background: ${COLORS.textPencil}; }
      .cache-signature {
        position: absolute;
        bottom: 10px;
        right: 15px;
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 0.9rem;
        color: ${COLORS.terracotta};
        transform: rotate(-5deg);
      }
      .cache-empty {
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: ${COLORS.textLight};
        padding: 0.5rem;
      }
    `;
    document.head.appendChild(style);
  }

  // ===========================================
  // RENDERING
  // ===========================================

  function createMatrixCells() {
    els.srcMatrix.innerHTML = '';
    els.dstMatrix.innerHTML = '';

    for (let idx = 0; idx < MATRIX_SIZE; idx++) {
      const i = Math.floor(idx / CONFIG.n);
      const j = idx % CONFIG.n;

      const srcCell = document.createElement('div');
      srcCell.id = `src-${idx}`;
      srcCell.className = 'cache-cell';
      srcCell.textContent = `${i},${j}`;
      els.srcMatrix.appendChild(srcCell);

      const dstCell = document.createElement('div');
      dstCell.id = `dst-${idx}`;
      dstCell.className = 'cache-cell';
      dstCell.textContent = '';
      els.dstMatrix.appendChild(dstCell);
    }
  }

  function renderCache() {
    els.l1Cache.innerHTML = '';

    if (state.l1Cache.length === 0) {
      els.l1Cache.innerHTML = '<div class="cache-empty">[ empty ]</div>';
      return;
    }

    state.l1Cache.forEach((line, idx) => {
      const lineDiv = document.createElement('div');
      lineDiv.className = `cache-line-group cache-line-${line.source}`;
      lineDiv.setAttribute('data-label', `${line.source}[${line.lineStart}..${line.lineStart + CONFIG.cacheLineSize - 1}]`);

      line.data.forEach(cellIdx => {
        if (cellIdx < MATRIX_SIZE) {
          const cell = document.createElement('div');
          const i = Math.floor(cellIdx / CONFIG.n);
          const j = cellIdx % CONFIG.n;
          cell.className = 'cache-line-cell';
          cell.textContent = `${i},${j}`;
          lineDiv.appendChild(cell);
        }
      });

      els.l1Cache.appendChild(lineDiv);
    });
  }

  function updateStats() {
    els.hitsDisplay.textContent = state.cacheHits;
    els.missesDisplay.textContent = state.cacheMisses;
    els.evictionsDisplay.textContent = state.evictions;
    els.timeDisplay.textContent = `${state.totalTime}s`;
  }

  function highlightCell(matrixEl, idx, type) {
    const cell = document.getElementById(`${matrixEl}-${idx}`);
    if (cell) {
      cell.classList.add('cache-cell-highlight', `cache-cell-${type}`);
    }
  }

  function resetCellHighlight(matrixEl, idx) {
    const cell = document.getElementById(`${matrixEl}-${idx}`);
    if (cell) {
      cell.classList.remove('cache-cell-highlight', 'cache-cell-reading', 'cache-cell-writing', 'cache-cell-loading');

      const lineStart = getCacheLineStart(idx);
      if (matrixEl === 'src' && state.srcCompleted.has(idx)) {
        cell.classList.add('cache-cell-done');
      } else if (matrixEl === 'dst' && state.dstCompleted.has(idx)) {
        cell.classList.add('cache-cell-done');
      } else if (isInCache(matrixEl, lineStart)) {
        cell.classList.add('cache-cell-cached');
      } else {
        cell.classList.remove('cache-cell-cached', 'cache-cell-done');
      }
    }
  }

  function updateAllCellHighlights() {
    // Update src cells
    for (let idx = 0; idx < MATRIX_SIZE; idx++) {
      const srcCell = document.getElementById(`src-${idx}`);
      const dstCell = document.getElementById(`dst-${idx}`);
      const lineStart = getCacheLineStart(idx);

      if (srcCell && !srcCell.classList.contains('cache-cell-highlight')) {
        srcCell.classList.remove('cache-cell-cached');
        if (state.srcCompleted.has(idx)) {
          srcCell.classList.add('cache-cell-done');
        } else if (isInCache('src', lineStart)) {
          srcCell.classList.add('cache-cell-cached');
        }
      }

      if (dstCell && !dstCell.classList.contains('cache-cell-highlight')) {
        dstCell.classList.remove('cache-cell-cached');
        if (state.dstCompleted.has(idx)) {
          dstCell.classList.add('cache-cell-done');
        } else if (isInCache('dst', lineStart)) {
          dstCell.classList.add('cache-cell-cached');
        }
      }
    }
  }

  // ===========================================
  // ANIMATION LOGIC
  // ===========================================

  async function executeStep() {
    if (state.finished) return;

    const srcIdx = state.i * CONFIG.n + state.j;
    const dstIdx = state.j * CONFIG.n + state.i;
    const srcCacheLine = getCacheLineStart(srcIdx);
    const dstCacheLine = getCacheLineStart(dstIdx);

    if (state.phase === 'read') {
      // PHASE 1: Read from src
      const result = accessCache('src', srcCacheLine);

      if (result.hit) {
        state.cacheHits++;
        state.totalTime += CONFIG.l1ReadTime;
        els.opDisplay.textContent = 'L1 READ';
        els.opDisplay.style.color = COLORS.sage;
      } else {
        state.cacheMisses++;
        state.totalTime += CONFIG.ramReadTime;
        els.opDisplay.textContent = 'RAM READ';
        els.opDisplay.style.color = COLORS.terracotta;
      }
      els.currentOp.innerHTML = `val = src[${state.i} * ${CONFIG.n} + ${state.j}]`;

      highlightCell('src', srcIdx, 'reading');
      renderCache();
      updateAllCellHighlights();
      updateStats();

      await delay(150);

      resetCellHighlight('src', srcIdx);
      state.srcCompleted.add(srcIdx);

      // Check if dst cache line is in cache
      if (isInCache('dst', dstCacheLine)) {
        state.phase = 'write';
      } else {
        state.phase = 'write-fetch';
      }

    } else if (state.phase === 'write-fetch') {
      // PHASE 2a: Write miss - must load dst cache line first (write-allocate)
      accessCache('dst', dstCacheLine);

      state.cacheMisses++;
      state.totalTime += CONFIG.ramReadTime;
      els.opDisplay.textContent = 'FETCH';
      els.opDisplay.style.color = COLORS.evict;

      els.currentOp.innerHTML = `<span style="color:${COLORS.evict}">fetch dst[${dstCacheLine}..${dstCacheLine + CONFIG.cacheLineSize - 1}]</span>`;

      highlightCell('dst', dstIdx, 'loading');
      renderCache();
      updateAllCellHighlights();
      updateStats();

      await delay(150);

      resetCellHighlight('dst', dstIdx);
      state.phase = 'write';

    } else {
      // PHASE 2b or 3: Write to dst (line is now in cache)
      // Touch the cache line to update LRU
      accessCache('dst', dstCacheLine);

      state.cacheHits++;
      state.totalTime += CONFIG.l1WriteTime;
      els.opDisplay.textContent = 'L1 WRITE';
      els.opDisplay.style.color = COLORS.sage;

      els.currentOp.innerHTML = `dst[${state.j} * ${CONFIG.n} + ${state.i}] = val`;

      highlightCell('dst', dstIdx, 'writing');

      const dstCell = document.getElementById(`dst-${dstIdx}`);
      if (dstCell) {
        dstCell.textContent = `${state.i},${state.j}`;
      }

      renderCache();
      updateAllCellHighlights();
      updateStats();

      await delay(150);

      resetCellHighlight('dst', dstIdx);
      state.dstCompleted.add(dstIdx);

      // Move to next element
      state.j++;
      if (state.j >= CONFIG.n) {
        state.j = 0;
        state.i++;
      }

      state.phase = 'read';

      // Check if done
      if (state.i >= CONFIG.n) {
        state.finished = true;
        state.playing = false;
        const total = state.cacheHits + state.cacheMisses;
        const efficiency = ((state.cacheHits / total) * 100).toFixed(1);
        els.currentOp.innerHTML = `<span style="color:${COLORS.sage}">// DONE — ${efficiency}% cache efficiency</span>`;
        els.opDisplay.textContent = 'DONE';
        updateButtons();
        return;
      }
    }
  }

  function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function updateButtons() {
    if (state.finished) {
      els.playBtn.textContent = 'PLAY';
      els.playBtn.classList.remove('playing');
      els.playBtn.disabled = true;
      els.stepBtn.disabled = true;
    } else if (state.playing) {
      els.playBtn.textContent = 'PAUSE';
      els.playBtn.classList.add('playing');
      els.playBtn.disabled = false;
      els.stepBtn.disabled = true;
    } else {
      els.playBtn.textContent = 'PLAY';
      els.playBtn.classList.remove('playing');
      els.playBtn.disabled = false;
      els.stepBtn.disabled = false;
    }
  }

  function togglePlay() {
    if (state.finished) return;

    state.playing = !state.playing;
    updateButtons();

    if (state.playing) {
      runAnimation();
    }
  }

  function doStep() {
    if (state.playing || state.finished) return;
    executeStep();
  }

  async function runAnimation() {
    while (state.playing && !state.finished) {
      await executeStep();
      if (state.playing && !state.finished) {
        await delay(400);
      }
    }
  }

  function reset() {
    state.playing = false;
    state.finished = false;
    state.i = 0;
    state.j = 0;
    state.phase = 'read';
    state.cacheHits = 0;
    state.cacheMisses = 0;
    state.evictions = 0;
    state.totalTime = 0;
    state.l1Cache = [];
    state.srcCompleted = new Set();
    state.dstCompleted = new Set();

    createMatrixCells();
    renderCache();
    updateStats();
    updateButtons();
    els.opDisplay.textContent = '—';
    els.currentOp.innerHTML = '<span class="cache-comment">// Waiting to start...</span>';
  }

  function init() {
    const container = document.getElementById('cache-animation-container');
    if (!container) return;

    createHTML(container);

    els = {
      playBtn: document.getElementById('cache-play-btn'),
      stepBtn: document.getElementById('cache-step-btn'),
      resetBtn: document.getElementById('cache-reset-btn'),
      hitsDisplay: document.getElementById('cache-hits'),
      missesDisplay: document.getElementById('cache-misses'),
      evictionsDisplay: document.getElementById('cache-evictions'),
      timeDisplay: document.getElementById('cache-time'),
      opDisplay: document.getElementById('cache-op'),
      currentOp: document.getElementById('cache-current-op'),
      l1Cache: document.getElementById('cache-l1'),
      srcMatrix: document.getElementById('cache-src'),
      dstMatrix: document.getElementById('cache-dst'),
    };

    createMatrixCells();

    els.playBtn.addEventListener('click', togglePlay);
    els.stepBtn.addEventListener('click', doStep);
    els.resetBtn.addEventListener('click', reset);

    renderCache();
    updateStats();
    updateButtons();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
