/**
 * Cache-Optimized Blocked Transpose Visualizer
 * Demonstrates cache-aware behavior during blocked matrix transpose
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
    blockSize: 4,            // Block size for tiled transpose
    cacheLineSize: 4,        // Units per cache line (e.g., 4 doubles = 32 bytes)
    l1CacheSize: 32,         // Total L1 cache units (8 cache lines)
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
    blockHighlight: 'rgba(107, 143, 113, 0.15)',
  };

  // Derived values
  const MATRIX_SIZE = CONFIG.n * CONFIG.n;
  const CACHE_LINES_IN_L1 = Math.floor(CONFIG.l1CacheSize / CONFIG.cacheLineSize);

  // State
  let state = {
    playing: false,
    finished: false,
    // Block iterators
    blockI: 0,        // Current block row
    blockJ: 0,        // Current block col
    // Within-block iterators
    bi: 0,            // Row within block
    bj: 0,            // Col within block
    phase: 'read',    // 'read', 'write-fetch', 'write'
    cacheHits: 0,
    cacheMisses: 0,
    evictions: 0,
    totalTime: 0,
    l1Cache: [],
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

  function findInCache(source, lineStart) {
    for (let i = 0; i < state.l1Cache.length; i++) {
      if (state.l1Cache[i].source === source && state.l1Cache[i].lineStart === lineStart) {
        return { index: i, entry: state.l1Cache[i] };
      }
    }
    return null;
  }

  function accessCache(source, lineStart) {
    const found = findInCache(source, lineStart);

    if (found) {
      state.l1Cache.splice(found.index, 1);
      state.l1Cache.push(found.entry);
      return { hit: true, evicted: null };
    }

    let evicted = null;

    if (state.l1Cache.length >= CACHE_LINES_IN_L1) {
      const evictedEntry = state.l1Cache.shift();
      evicted = { source: evictedEntry.source, lineStart: evictedEntry.lineStart };
      state.evictions++;
    }

    state.l1Cache.push({
      source: source,
      lineStart: lineStart,
      data: Array.from({ length: CONFIG.cacheLineSize }, (_, i) => lineStart + i)
    });

    return { hit: false, evicted: evicted };
  }

  function isInCache(source, lineStart) {
    return findInCache(source, lineStart) !== null;
  }

  // ===========================================
  // HTML GENERATION
  // ===========================================

  function createHTML(container) {
    container.innerHTML = `
      <svg class="blocked-crown" width="60" height="40" viewBox="0 0 60 40">
        <path d="M5 35 L15 10 L20 25 L30 5 L40 25 L45 10 L55 35" stroke="${COLORS.sage}" stroke-width="3" fill="none"/>
        <circle cx="15" cy="8" r="3" fill="${COLORS.mustard}"/>
        <circle cx="30" cy="3" r="3" fill="${COLORS.mustard}"/>
        <circle cx="45" cy="8" r="3" fill="${COLORS.mustard}"/>
      </svg>
      <h3 class="blocked-title">BLOCKED TRANSPOSE VISUALIZER</h3>
      <p class="blocked-subtitle">Watch how blocked/tiled transpose maximizes cache efficiency. Same write-allocate policy, dramatically better results.</p>
      <div class="blocked-controls">
        <button id="blocked-play-btn" class="blocked-btn blocked-btn-play">PLAY</button>
        <button id="blocked-step-btn" class="blocked-btn blocked-btn-step" disabled>STEP</button>
        <button id="blocked-reset-btn" class="blocked-btn blocked-btn-reset">RESET</button>
      </div>
      <div class="blocked-stats">
        <div class="blocked-stat">
          <div class="blocked-stat-value blocked-stat-hits" id="blocked-hits">0</div>
          <div class="blocked-stat-label">L1 HITS</div>
        </div>
        <div class="blocked-stat">
          <div class="blocked-stat-value blocked-stat-misses" id="blocked-misses">0</div>
          <div class="blocked-stat-label">L1 MISSES</div>
        </div>
        <div class="blocked-stat">
          <div class="blocked-stat-value blocked-stat-evictions" id="blocked-evictions">0</div>
          <div class="blocked-stat-label">EVICTIONS</div>
        </div>
        <div class="blocked-stat">
          <div class="blocked-stat-value blocked-stat-time" id="blocked-time">0s</div>
          <div class="blocked-stat-label">TOTAL TIME</div>
        </div>
        <div class="blocked-stat">
          <div class="blocked-stat-value blocked-stat-op" id="blocked-op">—</div>
          <div class="blocked-stat-label">OPERATION</div>
        </div>
      </div>
      <div class="blocked-block-info" id="blocked-block-info">
        <span class="blocked-comment">// Block: (0,0)</span>
      </div>
      <div class="blocked-current-op" id="blocked-current-op">
        <span class="blocked-comment">// Waiting to start...</span>
      </div>
      <div class="blocked-visual">
        <div class="blocked-section">
          <div class="blocked-section-title"><span class="blocked-dot blocked-dot-sage">●</span> L1 CACHE <span class="blocked-section-info">(${CONFIG.l1CacheSize} units / ${CACHE_LINES_IN_L1} lines, LRU)</span></div>
          <div class="blocked-l1" id="blocked-l1"></div>
        </div>
        <div class="blocked-matrices">
          <div class="blocked-section">
            <div class="blocked-section-title"><span class="blocked-dot blocked-dot-terracotta">●</span> SRC MATRIX <span class="blocked-section-info">(${CONFIG.n}×${CONFIG.n}, block=${CONFIG.blockSize})</span></div>
            <div class="blocked-matrix" id="blocked-src"></div>
          </div>
          <div class="blocked-section">
            <div class="blocked-section-title"><span class="blocked-dot blocked-dot-mustard">●</span> DST MATRIX <span class="blocked-section-info">(transposed)</span></div>
            <div class="blocked-matrix" id="blocked-dst"></div>
          </div>
        </div>
      </div>
      <div class="blocked-legend">
        <div class="blocked-legend-item"><span class="blocked-legend-box blocked-legend-block"></span> Current Block</div>
        <div class="blocked-legend-item"><span class="blocked-legend-box blocked-legend-reading"></span> Reading</div>
        <div class="blocked-legend-item"><span class="blocked-legend-box blocked-legend-loading"></span> Loading</div>
        <div class="blocked-legend-item"><span class="blocked-legend-box blocked-legend-writing"></span> Writing</div>
        <div class="blocked-legend-item"><span class="blocked-legend-box blocked-legend-cached"></span> In Cache</div>
        <div class="blocked-legend-item"><span class="blocked-legend-box blocked-legend-done"></span> Completed</div>
      </div>
      <div class="blocked-signature">BLOCK BY BLOCK ©</div>
    `;

    injectStyles();
  }

  function injectStyles() {
    if (document.getElementById('blocked-animation-styles')) return;

    const style = document.createElement('style');
    style.id = 'blocked-animation-styles';
    style.textContent = `
      #blocked-animation-container {
        margin: 2rem 0;
        padding: 1.5rem;
        background: ${COLORS.bgCream};
        border: 4px solid ${COLORS.textInk};
        position: relative;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }
      .blocked-crown {
        position: absolute;
        top: -20px;
        left: 20px;
      }
      .blocked-title {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.8rem;
        letter-spacing: 3px;
        color: ${COLORS.textInk};
        text-transform: uppercase;
        margin: 0 0 1rem 0;
        border-bottom: 3px solid ${COLORS.sage};
        padding-bottom: 0.5rem;
      }
      .blocked-subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: ${COLORS.textLight};
        margin-bottom: 1rem;
      }
      .blocked-controls {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
        align-items: center;
      }
      .blocked-btn {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.2rem;
        letter-spacing: 2px;
        padding: 0.5rem 1.5rem;
        border: 3px solid ${COLORS.textInk};
        cursor: pointer;
        text-transform: uppercase;
        transition: all 0.2s;
      }
      .blocked-btn:hover:not(:disabled) { transform: translateY(-2px); }
      .blocked-btn:disabled { opacity: 0.4; cursor: not-allowed; }
      .blocked-btn-play { background: ${COLORS.sage}; color: ${COLORS.bgCream}; }
      .blocked-btn-play.playing { background: ${COLORS.terracotta}; }
      .blocked-btn-step { background: ${COLORS.mustard}; color: ${COLORS.textInk}; }
      .blocked-btn-reset { background: ${COLORS.textPencil}; color: ${COLORS.bgCream}; }
      .blocked-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: ${COLORS.bgPaper};
        border: 2px solid ${COLORS.textInk};
      }
      .blocked-stat { text-align: center; }
      .blocked-stat-value {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.4rem;
      }
      .blocked-stat-hits { color: ${COLORS.sage}; }
      .blocked-stat-misses { color: ${COLORS.terracotta}; }
      .blocked-stat-evictions { color: ${COLORS.evict}; }
      .blocked-stat-time { color: ${COLORS.textInk}; }
      .blocked-stat-op { color: ${COLORS.mustard}; }
      .blocked-stat-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: ${COLORS.textLight};
      }
      .blocked-block-info {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.1rem;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: ${COLORS.sage};
        color: ${COLORS.bgCream};
        display: inline-block;
      }
      .blocked-current-op {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: ${COLORS.textInk};
        color: ${COLORS.bgCream};
        border-left: 4px solid ${COLORS.sage};
        white-space: pre-wrap;
        line-height: 1.6;
      }
      .blocked-comment { color: ${COLORS.mustard}; }
      .blocked-visual {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
      }
      .blocked-matrices {
        display: flex;
        flex-wrap: wrap;
        gap: 2rem;
      }
      .blocked-section-title {
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 1.2rem;
        letter-spacing: 2px;
        color: ${COLORS.textInk};
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      .blocked-section-info {
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: ${COLORS.textLight};
      }
      .blocked-dot-sage { color: ${COLORS.sage}; }
      .blocked-dot-terracotta { color: ${COLORS.terracotta}; }
      .blocked-dot-mustard { color: ${COLORS.mustard}; }
      .blocked-l1 {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        padding: 0.5rem;
        background: ${COLORS.bgPaper};
        border: 3px solid ${COLORS.sage};
        min-height: 60px;
      }
      .blocked-matrix {
        display: grid;
        gap: 1px;
        padding: 0.5rem;
        background: ${COLORS.bgPaper};
        grid-template-columns: repeat(${CONFIG.n}, 32px);
        width: fit-content;
      }
      #blocked-src { border: 3px solid ${COLORS.terracotta}; }
      #blocked-dst { border: 3px solid ${COLORS.mustard}; }
      .blocked-cell {
        width: 32px;
        height: 32px;
        background: ${COLORS.bgCream};
        border: 2px solid ${COLORS.textInk};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: ${COLORS.textPencil};
        transition: all 0.15s;
        box-sizing: border-box;
      }
      .blocked-cell-highlight { transform: scale(1.15); z-index: 10; }
      .blocked-cell-reading { background: ${COLORS.terracotta} !important; color: ${COLORS.bgCream} !important; }
      .blocked-cell-loading { background: ${COLORS.evict} !important; color: ${COLORS.bgCream} !important; }
      .blocked-cell-writing { background: ${COLORS.mustard} !important; color: ${COLORS.textInk} !important; }
      .blocked-cell-cached { background: rgba(107, 143, 113, 0.3); }
      .blocked-cell-done { background: ${COLORS.textPencil} !important; color: ${COLORS.bgCream} !important; }
      .blocked-cell-block { box-shadow: inset 0 0 0 2px ${COLORS.sage}; }
      .blocked-line-cell {
        width: 24px;
        height: 24px;
        background: ${COLORS.sage};
        border: 1px solid ${COLORS.textInk};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: ${COLORS.bgCream};
      }
      .blocked-line-group {
        display: flex;
        flex-wrap: nowrap;
        gap: 1px;
        padding: 4px;
        border: 2px solid;
        margin: 2px;
        position: relative;
      }
      .blocked-line-group::before {
        content: attr(data-label);
        position: absolute;
        top: -10px;
        left: 4px;
        font-family: 'Space Mono', monospace;
        font-size: 0.55rem;
        background: ${COLORS.bgPaper};
        padding: 0 2px;
      }
      .blocked-line-src { background: rgba(196, 93, 58, 0.2); border-color: ${COLORS.terracotta}; }
      .blocked-line-dst { background: rgba(212, 168, 75, 0.2); border-color: ${COLORS.mustard}; }
      .blocked-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 2px dashed ${COLORS.textInk};
      }
      .blocked-legend-item {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
      }
      .blocked-legend-box {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid ${COLORS.textInk};
      }
      .blocked-legend-block { background: ${COLORS.bgCream}; box-shadow: inset 0 0 0 2px ${COLORS.sage}; }
      .blocked-legend-reading { background: ${COLORS.terracotta}; }
      .blocked-legend-loading { background: ${COLORS.evict}; }
      .blocked-legend-writing { background: ${COLORS.mustard}; }
      .blocked-legend-cached { background: ${COLORS.sage}; }
      .blocked-legend-done { background: ${COLORS.textPencil}; }
      .blocked-signature {
        position: absolute;
        bottom: 10px;
        right: 15px;
        font-family: 'Bebas Neue', Impact, sans-serif;
        font-size: 0.9rem;
        color: ${COLORS.sage};
        transform: rotate(-5deg);
      }
      .blocked-empty {
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
      srcCell.id = `blocked-src-${idx}`;
      srcCell.className = 'blocked-cell';
      srcCell.textContent = `${i},${j}`;
      els.srcMatrix.appendChild(srcCell);

      const dstCell = document.createElement('div');
      dstCell.id = `blocked-dst-${idx}`;
      dstCell.className = 'blocked-cell';
      dstCell.textContent = '';
      els.dstMatrix.appendChild(dstCell);
    }
  }

  function renderCache() {
    els.l1Cache.innerHTML = '';

    if (state.l1Cache.length === 0) {
      els.l1Cache.innerHTML = '<div class="blocked-empty">[ empty ]</div>';
      return;
    }

    state.l1Cache.forEach((line) => {
      const lineDiv = document.createElement('div');
      lineDiv.className = `blocked-line-group blocked-line-${line.source}`;
      lineDiv.setAttribute('data-label', `${line.source}[${line.lineStart}..${line.lineStart + CONFIG.cacheLineSize - 1}]`);

      line.data.forEach(cellIdx => {
        if (cellIdx < MATRIX_SIZE) {
          const cell = document.createElement('div');
          const i = Math.floor(cellIdx / CONFIG.n);
          const j = cellIdx % CONFIG.n;
          cell.className = 'blocked-line-cell';
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
    const cell = document.getElementById(`blocked-${matrixEl}-${idx}`);
    if (cell) {
      cell.classList.add('blocked-cell-highlight', `blocked-cell-${type}`);
    }
  }

  function resetCellHighlight(matrixEl, idx) {
    const cell = document.getElementById(`blocked-${matrixEl}-${idx}`);
    if (cell) {
      cell.classList.remove('blocked-cell-highlight', 'blocked-cell-reading', 'blocked-cell-writing', 'blocked-cell-loading');

      const lineStart = getCacheLineStart(idx);
      if (matrixEl === 'src' && state.srcCompleted.has(idx)) {
        cell.classList.add('blocked-cell-done');
      } else if (matrixEl === 'dst' && state.dstCompleted.has(idx)) {
        cell.classList.add('blocked-cell-done');
      } else if (isInCache(matrixEl, lineStart)) {
        cell.classList.add('blocked-cell-cached');
      } else {
        cell.classList.remove('blocked-cell-cached', 'blocked-cell-done');
      }
    }
  }

  function updateBlockHighlights() {
    // Clear all block highlights first
    for (let idx = 0; idx < MATRIX_SIZE; idx++) {
      const srcCell = document.getElementById(`blocked-src-${idx}`);
      const dstCell = document.getElementById(`blocked-dst-${idx}`);
      if (srcCell) srcCell.classList.remove('blocked-cell-block');
      if (dstCell) dstCell.classList.remove('blocked-cell-block');
    }

    if (state.finished) return;

    // Highlight current block in both matrices
    const blockRowStart = state.blockI;
    const blockRowEnd = Math.min(state.blockI + CONFIG.blockSize, CONFIG.n);
    const blockColStart = state.blockJ;
    const blockColEnd = Math.min(state.blockJ + CONFIG.blockSize, CONFIG.n);

    for (let i = blockRowStart; i < blockRowEnd; i++) {
      for (let j = blockColStart; j < blockColEnd; j++) {
        const srcIdx = i * CONFIG.n + j;
        const dstIdx = j * CONFIG.n + i;
        const srcCell = document.getElementById(`blocked-src-${srcIdx}`);
        const dstCell = document.getElementById(`blocked-dst-${dstIdx}`);
        if (srcCell) srcCell.classList.add('blocked-cell-block');
        if (dstCell) dstCell.classList.add('blocked-cell-block');
      }
    }
  }

  function updateAllCellHighlights() {
    for (let idx = 0; idx < MATRIX_SIZE; idx++) {
      const srcCell = document.getElementById(`blocked-src-${idx}`);
      const dstCell = document.getElementById(`blocked-dst-${idx}`);
      const lineStart = getCacheLineStart(idx);

      if (srcCell && !srcCell.classList.contains('blocked-cell-highlight')) {
        srcCell.classList.remove('blocked-cell-cached');
        if (state.srcCompleted.has(idx)) {
          srcCell.classList.add('blocked-cell-done');
        } else if (isInCache('src', lineStart)) {
          srcCell.classList.add('blocked-cell-cached');
        }
      }

      if (dstCell && !dstCell.classList.contains('blocked-cell-highlight')) {
        dstCell.classList.remove('blocked-cell-cached');
        if (state.dstCompleted.has(idx)) {
          dstCell.classList.add('blocked-cell-done');
        } else if (isInCache('dst', lineStart)) {
          dstCell.classList.add('blocked-cell-cached');
        }
      }
    }
  }

  // ===========================================
  // ANIMATION LOGIC
  // ===========================================

  function getCurrentIndices() {
    const i = state.blockI + state.bi;
    const j = state.blockJ + state.bj;
    return { i, j };
  }

  function advanceToNext() {
    state.bj++;
    const blockColEnd = Math.min(state.blockJ + CONFIG.blockSize, CONFIG.n);
    const blockRowEnd = Math.min(state.blockI + CONFIG.blockSize, CONFIG.n);

    if (state.blockJ + state.bj >= blockColEnd) {
      state.bj = 0;
      state.bi++;
      if (state.blockI + state.bi >= blockRowEnd) {
        // Move to next block
        state.bi = 0;
        state.bj = 0;
        state.blockJ += CONFIG.blockSize;
        if (state.blockJ >= CONFIG.n) {
          state.blockJ = 0;
          state.blockI += CONFIG.blockSize;
        }
      }
    }
  }

  function isFinished() {
    return state.blockI >= CONFIG.n;
  }

  async function executeStep() {
    if (state.finished) return;

    const { i, j } = getCurrentIndices();
    const srcIdx = i * CONFIG.n + j;
    const dstIdx = j * CONFIG.n + i;
    const srcCacheLine = getCacheLineStart(srcIdx);
    const dstCacheLine = getCacheLineStart(dstIdx);

    // Update block info display
    els.blockInfo.innerHTML = `BLOCK: (${Math.floor(state.blockI / CONFIG.blockSize)}, ${Math.floor(state.blockJ / CONFIG.blockSize)}) — rows [${state.blockI}..${Math.min(state.blockI + CONFIG.blockSize, CONFIG.n) - 1}], cols [${state.blockJ}..${Math.min(state.blockJ + CONFIG.blockSize, CONFIG.n) - 1}]`;

    if (state.phase === 'read') {
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
      els.currentOp.innerHTML = `val = src[${i} * ${CONFIG.n} + ${j}]`;

      highlightCell('src', srcIdx, 'reading');
      renderCache();
      updateBlockHighlights();
      updateAllCellHighlights();
      updateStats();

      await delay(150);

      resetCellHighlight('src', srcIdx);
      state.srcCompleted.add(srcIdx);

      if (isInCache('dst', dstCacheLine)) {
        state.phase = 'write';
      } else {
        state.phase = 'write-fetch';
      }

    } else if (state.phase === 'write-fetch') {
      accessCache('dst', dstCacheLine);

      state.cacheMisses++;
      state.totalTime += CONFIG.ramReadTime;
      els.opDisplay.textContent = 'FETCH';
      els.opDisplay.style.color = COLORS.evict;

      els.currentOp.innerHTML = `<span style="color:${COLORS.evict}">fetch dst[${dstCacheLine}..${dstCacheLine + CONFIG.cacheLineSize - 1}]</span>`;

      highlightCell('dst', dstIdx, 'loading');
      renderCache();
      updateBlockHighlights();
      updateAllCellHighlights();
      updateStats();

      await delay(150);

      resetCellHighlight('dst', dstIdx);
      state.phase = 'write';

    } else {
      accessCache('dst', dstCacheLine);

      state.cacheHits++;
      state.totalTime += CONFIG.l1WriteTime;
      els.opDisplay.textContent = 'L1 WRITE';
      els.opDisplay.style.color = COLORS.sage;

      els.currentOp.innerHTML = `dst[${j} * ${CONFIG.n} + ${i}] = val`;

      highlightCell('dst', dstIdx, 'writing');

      const dstCell = document.getElementById(`blocked-dst-${dstIdx}`);
      if (dstCell) {
        dstCell.textContent = `${i},${j}`;
      }

      renderCache();
      updateBlockHighlights();
      updateAllCellHighlights();
      updateStats();

      await delay(150);

      resetCellHighlight('dst', dstIdx);
      state.dstCompleted.add(dstIdx);

      advanceToNext();
      state.phase = 'read';

      if (isFinished()) {
        state.finished = true;
        state.playing = false;
        const total = state.cacheHits + state.cacheMisses;
        const efficiency = ((state.cacheHits / total) * 100).toFixed(1);
        els.currentOp.innerHTML = `<span style="color:${COLORS.sage}">// DONE — ${efficiency}% cache efficiency</span>`;
        els.opDisplay.textContent = 'DONE';
        els.blockInfo.innerHTML = `COMPLETE — ${efficiency}% efficiency vs naive transpose!`;
        updateBlockHighlights();
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
        await delay(300);
      }
    }
  }

  function reset() {
    state.playing = false;
    state.finished = false;
    state.blockI = 0;
    state.blockJ = 0;
    state.bi = 0;
    state.bj = 0;
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
    updateBlockHighlights();
    els.opDisplay.textContent = '—';
    els.blockInfo.innerHTML = 'BLOCK: (0, 0)';
    els.currentOp.innerHTML = '<span class="blocked-comment">// Waiting to start...</span>';
  }

  function init() {
    const container = document.getElementById('blocked-animation-container');
    if (!container) return;

    createHTML(container);

    els = {
      playBtn: document.getElementById('blocked-play-btn'),
      stepBtn: document.getElementById('blocked-step-btn'),
      resetBtn: document.getElementById('blocked-reset-btn'),
      hitsDisplay: document.getElementById('blocked-hits'),
      missesDisplay: document.getElementById('blocked-misses'),
      evictionsDisplay: document.getElementById('blocked-evictions'),
      timeDisplay: document.getElementById('blocked-time'),
      opDisplay: document.getElementById('blocked-op'),
      blockInfo: document.getElementById('blocked-block-info'),
      currentOp: document.getElementById('blocked-current-op'),
      l1Cache: document.getElementById('blocked-l1'),
      srcMatrix: document.getElementById('blocked-src'),
      dstMatrix: document.getElementById('blocked-dst'),
    };

    createMatrixCells();

    els.playBtn.addEventListener('click', togglePlay);
    els.stepBtn.addEventListener('click', doStep);
    els.resetBtn.addEventListener('click', reset);

    renderCache();
    updateStats();
    updateButtons();
    updateBlockHighlights();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
