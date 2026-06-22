// Backend API base. Resolution order (matches entitlements.js / auth.js):
//   1. ?api=<url> query override — point a separately-served frontend at a backend.
//   2. Same-origin when served over http(s) — main.py mounts this frontend at "/",
//      so opening http://localhost:8000/templates.html talks to localhost:8000.
//      This is what makes the local /unmix/custom endpoint reachable instead of prod.
//   3. Deployed default — only when opened from disk (file://) without an override.
const API_BASE =
   (new URLSearchParams(location.search).get('api') || '').replace(/\/$/, '') ||
   (location.protocol === 'http:' || location.protocol === 'https:'
      ? location.origin
      : 'https://alchromaticdemo.up.railway.app');

// Palette presets (filled from colors.json by loadPalettePresets()).
let palettePresetsData = null;

// Load the named palette presets and populate the palette dropdown.
// Prefer the embedded global (works under file://, where fetch() is blocked);
// fall back to fetching colors.json when served over http(s).
async function loadPalettePresets() {
    if (window.PALETTE_PRESETS && typeof window.PALETTE_PRESETS === 'object') {
        palettePresetsData = window.PALETTE_PRESETS;
    } else {
        try {
            const response = await fetch('colors.json');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            palettePresetsData = await response.json();
        } catch (error) {
            console.warn('Could not load colors.json palette presets:', error.message);
            return;
        }
    }
    populateTrycolorsPaletteSelector();
}

function buildMixDotGrid(recipe, rows, cols, maxDots) {
    if (!recipe || recipe.length === 0) return '';
    rows = rows || 10;
    cols = cols || 10;
    const totalCells = rows * cols;
    // If maxDots is 0 or not set or greater than totalCells, fill all
    const fillCount = (!maxDots || maxDots <= 0 || maxDots > totalCells) ? totalCells : maxDots;

    // Calculate how many dots each color gets out of fillCount
    const dots = [];
    let assigned = 0;
    recipe.forEach((item, idx) => {
        const pct = parseFloat(item.percentage) || 0;
        const count = idx === recipe.length - 1
            ? (fillCount - assigned)
            : Math.round(pct / 100 * fillCount);
        dots.push({ hex: item.hex, count: Math.max(0, count) });
        assigned += Math.max(0, count);
    });
    // Build flat array: colored dots first, then empty
    const cells = [];
    dots.forEach(d => {
        for (let i = 0; i < d.count; i++) cells.push(d.hex);
    });
    // Trim to fillCount
    if (cells.length > fillCount) cells.length = fillCount;
    // Pad with null for unfilled cells
    while (cells.length < totalCells) cells.push(null);

    const filledLabel = fillCount < totalCells ? ` (${fillCount} of ${totalCells} filled)` : '';
    let html = `<div class="mix-grid-wrap"><div class="mix-grid-label">Mix Proportions${filledLabel}</div><div class="mix-grid" style="grid-template-columns: repeat(${cols}, 1fr);">`;
    for (let i = 0; i < totalCells; i++) {
        if (cells[i]) {
            html += `<div class="mix-grid-dot" style="background:${cells[i]}" title="${cells[i]}"></div>`;
        } else {
            html += `<div class="mix-grid-dot mix-grid-dot-empty" title="empty"></div>`;
        }
    }
    html += '</div></div>';
    return html;
}

// Squarified treemap layout algorithm (Bruls, Huizing, van Wijk)
function squarify(items, x, y, w, h) {
    const rects = [];
    if (!items || items.length === 0) return rects;
    const totalValue = items.reduce((s, it) => s + it.value, 0);
    if (totalValue <= 0) return rects;

    // Normalize values to fill the area
    const normalized = items.map(it => ({ ...it, area: (it.value / totalValue) * w * h }));

    function layoutRow(row, rowItems, rx, ry, rw, rh, isVertical) {
        const rowArea = rowItems.reduce((s, it) => s + it.area, 0);
        let offset = 0;
        rowItems.forEach(it => {
            const ratio = it.area / rowArea;
            if (isVertical) {
                const sliceW = rowArea / rh;
                const sliceH = rh * ratio;
                rects.push({ ...it, x: rx, y: ry + offset, w: sliceW, h: sliceH });
                offset += sliceH;
            } else {
                const sliceH = rowArea / rw;
                const sliceW = rw * ratio;
                rects.push({ ...it, x: rx + offset, y: ry, w: sliceW, h: sliceH });
                offset += sliceW;
            }
        });
        return rects;
    }

    function worst(row, sideLen) {
        const rowArea = row.reduce((s, it) => s + it.area, 0);
        const s2 = sideLen * sideLen;
        let maxR = 0;
        row.forEach(it => {
            const r = Math.max((s2 * it.area) / (rowArea * rowArea), (rowArea * rowArea) / (s2 * it.area));
            if (r > maxR) maxR = r;
        });
        return maxR;
    }

    function doLayout(items, cx, cy, cw, ch) {
        if (items.length === 0) return;
        if (items.length === 1) {
            rects.push({ ...items[0], x: cx, y: cy, w: cw, h: ch });
            return;
        }
        const isVertical = ch > cw;
        const sideLen = isVertical ? cw : ch;
        let row = [items[0]];
        let remaining = items.slice(1);
        let bestWorst = worst(row, sideLen);

        while (remaining.length > 0) {
            const candidate = [...row, remaining[0]];
            const cw2 = worst(candidate, sideLen);
            if (cw2 <= bestWorst) {
                row = candidate;
                remaining = remaining.slice(1);
                bestWorst = cw2;
            } else {
                break;
            }
        }

        const rowArea = row.reduce((s, it) => s + it.area, 0);
        if (isVertical) {
            const sliceW = rowArea / ch;
            layoutRow(rects, row, cx, cy, cw, ch, true);
            doLayout(remaining, cx + sliceW, cy, cw - sliceW, ch);
        } else {
            const sliceH = rowArea / cw;
            layoutRow(rects, row, cx, cy, cw, ch, false);
            doLayout(remaining, cx, cy + sliceH, cw, ch - sliceH);
        }
    }

    // Sort descending by area for better squarification
    normalized.sort((a, b) => b.area - a.area);
    doLayout(normalized, x, y, w, h);
    return rects;
}

// Build a JSON-style treemap visualization from recipe data
function buildMixTreemap(recipe) {
    if (!recipe || recipe.length === 0) return '';
    const svgW = 300, svgH = 300, pad = 2;

    const items = recipe.map(it => ({
        hex: it.hex,
        name: it.name || it.hex,
        percentage: parseFloat(it.percentage) || 0,
        parts: it.parts,
        value: parseFloat(it.percentage) || 0
    })).filter(it => it.value > 0);

    if (items.length === 0) return '';

    const rects = squarify(items, 0, 0, svgW, svgH);

    // Determine text color based on background luminance
    function textColor(hex) {
        const c = hex.replace('#', '');
        const r = parseInt(c.substring(0, 2), 16);
        const g = parseInt(c.substring(2, 4), 16);
        const b = parseInt(c.substring(4, 6), 16);
        return (r * 0.299 + g * 0.587 + b * 0.114) > 150 ? '#222' : '#fff';
    }

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" style="font-family: Arial, sans-serif;">`;

    rects.forEach(r => {
        const rx = r.x + pad / 2;
        const ry = r.y + pad / 2;
        const rw = Math.max(0, r.w - pad);
        const rh = Math.max(0, r.h - pad);
        const tc = textColor(r.hex);
        const pct = r.percentage.toFixed(1) + '%';
        const label = r.name.length > 14 ? r.name.substring(0, 12) + '…' : r.name;

        svg += `<g class="treemap-cell">`;
        svg += `<rect x="${rx.toFixed(1)}" y="${ry.toFixed(1)}" width="${rw.toFixed(1)}" height="${rh.toFixed(1)}" fill="${r.hex}" rx="3" stroke="rgba(0,0,0,0.15)" stroke-width="1"/>`;

        // Only show labels if cell is big enough
        if (rw > 35 && rh > 28) {
            const fontSize = Math.min(12, Math.max(8, rw / 8));
            const cx = rx + rw / 2;
            const cy = ry + rh / 2;
            svg += `<text x="${cx.toFixed(1)}" y="${(cy - fontSize * 0.3).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${fontSize.toFixed(1)}px" font-weight="600">${label}</text>`;
            svg += `<text x="${cx.toFixed(1)}" y="${(cy + fontSize * 0.9).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${(fontSize * 0.85).toFixed(1)}px" opacity="0.85">${pct}</text>`;
            if (rh > 45 && r.parts !== undefined) {
                svg += `<text x="${cx.toFixed(1)}" y="${(cy + fontSize * 2).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${(fontSize * 0.75).toFixed(1)}px" opacity="0.65">${r.parts} parts</text>`;
            }
        } else if (rw > 20 && rh > 16) {
            const fontSize = Math.max(7, Math.min(9, rw / 6));
            svg += `<text x="${(rx + rw / 2).toFixed(1)}" y="${(ry + rh / 2 + fontSize / 3).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${fontSize.toFixed(1)}px">${pct}</text>`;
        }

        svg += `<title>${r.name}\n${r.hex}\n${pct}${r.parts !== undefined ? '\n' + r.parts + ' parts' : ''}</title>`;
        svg += `</g>`;
    });

    svg += `</svg>`;

    return `<div class="mix-treemap-wrap"><div class="mix-treemap-label">Mix Proportions — Treemap</div><div class="mix-treemap-container">${svg}</div></div>`;
}

// Build concentric circles visualization — single diagram. Each color is drawn
// as a FULL CIRCLE (not a ring) sized by its percentage. The largest color is
// bottom-tangent to the 100% reference; smaller colors are top-tangent so they
// don't get hidden behind larger ones. All circles share the same 100% outline.
function buildMixConcentricClock(recipe) {
    if (!recipe || recipe.length === 0) return '';

    const items = recipe.map(it => ({
        hex: it.hex,
        name: it.name || it.hex,
        percentage: parseFloat(it.percentage) || 0,
        parts: it.parts
    })).filter(it => it.percentage > 0);

    if (items.length === 0) return '';

    // Sort descending (largest first)
    items.sort((a, b) => b.percentage - a.percentage);

    function textColor(hex) {
        const c = hex.replace('#', '');
        const r = parseInt(c.substring(0, 2), 16);
        const g = parseInt(c.substring(2, 4), 16);
        const b = parseInt(c.substring(4, 6), 16);
        return (r * 0.299 + g * 0.587 + b * 0.114) > 150 ? '#222' : '#fff';
    }

    const maxR = 150;               // radius of the 100% reference
    const diameter = maxR * 2;
    const pad = 28;
    const rightGuideW = 56;          // room on right for guide-line % labels
    // Legend goes underneath
    const legendCols = Math.min(items.length, 3);
    const legendRows = Math.ceil(items.length / legendCols);
    const legendRowH = 30;
    const legendH = 24 + legendRows * legendRowH;
    const diagramW = diameter + pad * 2 + rightGuideW;
    const svgW = diagramW;
    const svgH = diameter + pad * 2 + legendH;
    const cx = pad + maxR;
    const topY = pad;
    const bottomY = pad + diameter;

    // On-dark palette
    const lineColor = '#ffffff';
    const refStroke = '#cfe0f5';
    const labelColor = '#ffffff';
    const subLabelColor = '#cfe0f5';

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" style="font-family: Arial, sans-serif;">`;

    // Concentric reference outlines (100%, 80%, 60%, 40%, 20%, 10%) — bottom-tangent
    const refLevels = [100, 80, 60, 40, 20, 10];
    refLevels.forEach(pct => {
        const r = maxR * (pct / 100);
        const rcy = bottomY - r;
        const isOuter = (pct === 100);
        svg += `<circle cx="${cx}" cy="${rcy.toFixed(1)}" r="${r.toFixed(1)}" fill="none" stroke="${refStroke}" stroke-width="${isOuter ? 1.8 : 1}" ${isOuter ? '' : 'stroke-dasharray="5,4"'} opacity="${isOuter ? 0.85 : 0.45}"/>`;
    });

    // Horizontal guide lines at 10% / 50% / 100%
    const guides = [
        { pct: 10,  style: 'dashed' },
        { pct: 50,  style: 'solid' },
        { pct: 100, style: 'solid' }
    ];
    guides.forEach(g => {
        const lineY = bottomY - (g.pct / 100) * diameter;
        const isDashed = g.style === 'dashed';
        svg += `<line x1="0" y1="${lineY.toFixed(1)}" x2="${diagramW}" y2="${lineY.toFixed(1)}" stroke="${lineColor}" stroke-width="${isDashed ? 1.2 : 2}" ${isDashed ? 'stroke-dasharray="6,4"' : ''} opacity="0.9"/>`;
        svg += `<text x="${(diagramW - 4).toFixed(1)}" y="${(lineY + 4).toFixed(1)}" text-anchor="end" fill="${lineColor}" font-size="11px" font-weight="700">${g.pct}%</text>`;
    });
    // Baseline
    svg += `<line x1="0" y1="${bottomY}" x2="${diagramW}" y2="${bottomY}" stroke="${lineColor}" stroke-width="2" opacity="0.7"/>`;

    // Draw each color as a FULL filled circle sized to its percentage.
    // Stack circles tangent vertically along the central axis so they never overlap.
    // Largest sits at the bottom; each subsequent circle sits directly on top of the previous.
    const total = items.reduce((s, it) => s + it.percentage, 0) || 100;
    let stackedBottom = bottomY; // bottom y of the next circle to place
    items.forEach((item, idx) => {
        const r = maxR * (item.percentage / total);
        const cyCircle = stackedBottom - r;
        stackedBottom = stackedBottom - 2 * r; // next circle sits on top of this one
        const tc = textColor(item.hex);
        const pct = item.percentage.toFixed(1) + '%';
        const label = item.name.length > 12 ? item.name.substring(0, 10) + '…' : item.name;

        svg += `<circle cx="${cx}" cy="${cyCircle.toFixed(1)}" r="${r.toFixed(1)}" fill="${item.hex}" stroke="#ffffff" stroke-width="1.5"/>`;

        if (r > 22) {
            const fontSize = Math.min(13, Math.max(9, r / 3.5));
            svg += `<text x="${cx}" y="${(cyCircle - fontSize * 0.2).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${fontSize.toFixed(1)}px" font-weight="700">${label}</text>`;
            svg += `<text x="${cx}" y="${(cyCircle + fontSize * 0.95).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${(fontSize * 0.85).toFixed(1)}px" opacity="0.9">${pct}</text>`;
        } else if (r > 10) {
            const fontSize = Math.max(8, r / 2.5);
            svg += `<text x="${cx}" y="${(cyCircle + fontSize * 0.35).toFixed(1)}" text-anchor="middle" fill="${tc}" font-size="${fontSize.toFixed(1)}px" font-weight="700">${pct}</text>`;
        }

        svg += `<title>${item.name}\n${item.hex}\n${pct}${item.parts !== undefined ? '\n' + item.parts + ' parts' : ''}</title>`;
    });

    // Legend underneath — grid layout, centered
    const legendTop = pad + diameter + 18;
    svg += `<text x="${(svgW / 2).toFixed(1)}" y="${legendTop}" text-anchor="middle" fill="${labelColor}" font-size="12px" font-weight="700">Legend</text>`;
    const colW = svgW / legendCols;
    items.forEach((item, idx) => {
        const col = idx % legendCols;
        const row = Math.floor(idx / legendCols);
        const ex = col * colW + 14;
        const ey = legendTop + 12 + row * legendRowH;
        const label = item.name.length > 18 ? item.name.substring(0, 16) + '…' : item.name;
        svg += `<rect x="${ex}" y="${ey}" width="14" height="14" fill="${item.hex}" stroke="#ffffff" stroke-width="0.8"/>`;
        svg += `<text x="${ex + 20}" y="${(ey + 10).toFixed(1)}" fill="${labelColor}" font-size="11px" font-weight="600">${label}</text>`;
        svg += `<text x="${ex + 20}" y="${(ey + 22).toFixed(1)}" fill="${subLabelColor}" font-size="10px">${item.hex} · ${item.percentage.toFixed(1)}%</text>`;
    });

    svg += `</svg>`;

    return `<div class="mix-concentric-wrap"><div class="mix-concentric-label">Mix Proportions — Concentric Circles (adds up to 100%)</div><div class="mix-concentric-container">${svg}</div></div>`;
}


// Returns the appropriate visualization HTML based on selected mode
function buildMixVisualization(recipe, rows, cols, maxDots) {
    const mode = document.querySelector('.viz-mode-btn.active');
    const vizType = mode ? mode.getAttribute('data-viz') : 'dotgrid';
    if (vizType === 'treemap') return buildMixTreemap(recipe);
    if (vizType === 'concentric') return buildMixConcentricClock(recipe);
    return buildMixDotGrid(recipe, rows, cols, maxDots);
}

// Store last recipe data for re-rendering on viz mode switch
let _lastRecipeData = null;
let _lastTrycolorsData = null;
let _lastTrycolorsMaxParts = null;
let _lastVersionedUnmix = null;   // last /version/unmix response shown (with _selIdx)
let _versionedSortMode = 'match'; // 'match' = highest match first | 'reliable' = model risk-adjusted order

function renderRecipeResults(data) {
    _lastRecipeData = data;
    if (data.error) {
        recipeResults.innerHTML = `<div class="recipe-error">⚠️ ${data.error}</div>`;
        return;
    }
    
    if (!data.recipe || data.recipe.length === 0) {
        recipeResults.innerHTML = `<div class="recipe-empty">No recipe found for this color</div>`;
        return;
    }
    
    const matchClass = getRecipeMatchClass(data.match_percentage);
    
    let html = `
        <div class="recipe-header">
            <div class="recipe-header-left">
                <div class="recipe-target-swatch" style="background: ${data.target_color}" title="Target"></div>
                <div class="recipe-arrow">→</div>
                <div class="recipe-result-swatch" style="background: ${data.result_color}" title="Result"></div>
            </div>
            <div class="recipe-match-info">
                <div class="recipe-match-value ${matchClass}">${data.match_percentage}%</div>
                <div class="recipe-match-label">match</div>
                <div class="recipe-method">${formatMixMethod(data.mix_method)}</div>
            </div>
        </div>
        <div class="recipe-components">
    `;
    
    data.recipe.forEach(comp => {
        const displayName = comp.name || comp.hex;
        html += `
            <div class="recipe-component">
                <div class="recipe-component-swatch" style="background: ${comp.hex}"></div>
                <div class="recipe-component-info">
                    <div class="recipe-component-name" title="${displayName}">${displayName}</div>
                    <div class="recipe-component-hex">${comp.hex}</div>
                </div>
                <div class="recipe-component-amount">
                    <div class="recipe-component-parts">${comp.parts}</div>
                    <div class="recipe-component-parts-label">parts</div>
                    <div class="recipe-component-percent">${comp.percentage}%</div>
                </div>
            </div>
        `;
    });
    
    html += `</div>`;
    html += buildMixVisualization(data.recipe, parseInt(document.getElementById('gridRowsInput').value) || 10, parseInt(document.getElementById('gridColsInput').value) || 10, parseInt(document.getElementById('gridMaxDotsInput').value) || 0);
    html += `<div class="recipe-total">Total: ${data.total_parts} parts · ΔE: ${data.delta_e}</div>`;

    recipeResults.innerHTML = html;
}

// Build palette with names for the unmix request
function buildPaletteWithNames(hexColors) {
    // Try to get names from the loaded palette preset
    const select = document.getElementById('palettePresetSelect');
    const selectedPalette = select.value;
    
    if (selectedPalette && palettePresetsData && palettePresetsData[selectedPalette]) {
        const colorArray = palettePresetsData[selectedPalette];
        const paletteMap = new Map();
        
        // Build map of hex -> name
        colorArray.forEach((rgb, idx) => {
            const r = Math.round(rgb[0]);
            const g = Math.round(rgb[1]);
            const b = Math.round(rgb[2]);
            const hex = rgbToHex(r, g, b).toUpperCase();
            paletteMap.set(hex, `${selectedPalette} #${idx + 1}`);
        });
        
        // Return palette with names where available
        return hexColors.map(hex => ({
            hex: hex,
            name: paletteMap.get(hex.toUpperCase()) || null
        }));
    }
    
    // No preset loaded, return without names
    return hexColors.map(hex => ({ hex: hex, name: null }));
}

// Visualization mode toggle (Dot Matrix / JSON Treemap / Concentric Circles)
document.querySelectorAll('.viz-mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.viz-mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const settingsPanel = document.getElementById('dotGridSettingsPanel');
        if (settingsPanel) {
            settingsPanel.style.display = btn.getAttribute('data-viz') === 'dotgrid' ? '' : 'none';
        }
        // Re-render existing results with the new visualization mode
        if (_lastRecipeData) renderRecipeResults(_lastRecipeData);
        if (_lastVersionedUnmix) renderVersionedUnmix(_lastVersionedUnmix);
        else if (_lastTrycolorsData) renderTrycolorsResultsFromUnmix(_lastTrycolorsData, _lastTrycolorsMaxParts);
    });
});

// Dot grid settings: re-render visualization when rows/columns/maxDots change
function _reRenderGridSettings() {
    if (_lastRecipeData) renderRecipeResults(_lastRecipeData);
    if (_lastVersionedUnmix) renderVersionedUnmix(_lastVersionedUnmix);
    else if (_lastTrycolorsData) renderTrycolorsResultsFromUnmix(_lastTrycolorsData, _lastTrycolorsMaxParts);
}
['gridRowsInput', 'gridColsInput', 'gridMaxDotsInput'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', _reRenderGridSettings);
});

const trycolorsInput = document.getElementById('trycolorsInput');
const trycolorsPreview = document.getElementById('trycolorsPreview');
const trycolorsBtn = document.getElementById('trycolorsBtn');
const trycolorsResults = document.getElementById('trycolorsResults');
const trycolorsPaletteGrid = document.getElementById('trycolorsPaletteGrid');

// Default palette data (will be fetched from API)
let defaultPaletteData = [];
// Track silenced/excluded colors (by hex)
let silencedColors = new Set();

// Update preview as user types
trycolorsInput.addEventListener('input', () => {
    let hex = trycolorsInput.value.trim().toUpperCase();
    if (!hex.startsWith('#')) hex = '#' + hex;
    
    // Validate hex
    if (/^#[0-9A-F]{6}$/i.test(hex)) {
        trycolorsPreview.style.background = hex;
        trycolorsBtn.disabled = false;
    } else if (/^#[0-9A-F]{3}$/i.test(hex)) {
        // Expand 3-char hex
        const expanded = '#' + hex[1] + hex[1] + hex[2] + hex[2] + hex[3] + hex[3];
        trycolorsPreview.style.background = expanded;
        trycolorsBtn.disabled = false;
    } else {
        trycolorsPreview.style.background = '#333';
        trycolorsBtn.disabled = hex.replace('#', '').length < 3;
    }
});

// Note: renderTrycolorsResults removed - now using renderTrycolorsResultsFromUnmix which uses same /unmix endpoint as main section

// Copy formatted output to clipboard
function copyTrycolorsOutput() {
    const output = document.getElementById('trycolorsOutput');
    if (output) {
        navigator.clipboard.writeText(output.textContent).then(() => {
            const btn = document.querySelector('.trycolors-copy-btn');
            const original = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => btn.textContent = original, 1500);
        });
    }
}

// Track current palette source
let currentTrycolorsPaletteSource = 'default';

// Fetch default palette and populate grid
async function loadDefaultPalette() {
    try {
        const response = await fetch(`${API_BASE}/unmix/palette`);
        if (response.ok) {
            const data = await response.json();
            defaultPaletteData = data.palette;
            currentTrycolorsPaletteSource = 'default';
            silencedColors.clear(); // Clear silenced when loading new palette
            renderPaletteGrid();
        }
    } catch (error) {
        console.log('Could not load default palette:', error);
    }
}

// Populate the trycolors palette selector with presets
function populateTrycolorsPaletteSelector() {
    const select = document.getElementById('trycolorsPaletteSelect');
    if (!select || !palettePresetsData) return;
    
    // Clear existing options except default
    select.innerHTML = '<option value="default">Default Artist Palette (20 colors)</option>';
    
    // Add presets from colors.json
    Object.keys(palettePresetsData).forEach(paletteName => {
        const colorCount = palettePresetsData[paletteName].length;
        const option = document.createElement('option');
        option.value = paletteName;
        option.textContent = `${paletteName} (${colorCount} colors)`;
        select.appendChild(option);
    });

    // Add the user's editable Color Library (if any) — see color-library.js.
    if (window.ColorLibrary && window.ColorLibrary.size() > 0) {
        const opt = document.createElement('option');
        opt.value = '__mycolors__';
        opt.textContent = `★ My Colors (${window.ColorLibrary.size()} colors)`;
        select.appendChild(opt);
    }

    // Also list the user's other saved palettes from their account (DB).
    if (window.ColorLibrary && typeof window.ColorLibrary.appendServerOptions === 'function') {
        window.ColorLibrary.appendServerOptions(select);
    }
}

// Push an arbitrary [{hex, name}] palette into the Trycolors unmixer. Used by the
// Color Library editor so user-named colors flow into the unmix request.
window.applyUnmixerPalette = function (colors, sourceName) {
    if (!Array.isArray(colors)) return;
    defaultPaletteData = colors.map(c => ({ hex: c.hex, name: c.name || null }));
    currentTrycolorsPaletteSource = sourceName || 'custom';
    silencedColors.clear();
    renderPaletteGrid();
};

// Read the palette currently loaded in the unmixer (used by the Color Library "Import").
window.getUnmixerPalette = function () {
    return (defaultPaletteData || []).map(c => ({ hex: c.hex, name: c.name || null }));
};

// Load a preset palette into trycolors
function loadTrycolorsPalette(paletteName) {
    if (paletteName === 'default') {
        loadDefaultPalette();
        return;
    }

    // User's editable Color Library (names + RGB) — see color-library.js.
    if (paletteName === '__mycolors__' && window.ColorLibrary) {
        window.applyUnmixerPalette(window.ColorLibrary.asPalette(), '__mycolors__');
        return;
    }

    // A server-saved palette chosen from the dropdown (see color-library.js).
    if (paletteName && paletteName.indexOf('__srvpal__:') === 0 && window.ColorLibrary) {
        const sp = window.ColorLibrary.getServerPalette(paletteName.split(':')[1]);
        if (sp) {
            window.applyUnmixerPalette((sp.colors || []).map(c => ({ hex: c.hex, name: c.name || null })), '__srvpal__');
            return;
        }
    }

    if (!palettePresetsData || !palettePresetsData[paletteName]) {
        console.log('Palette not found:', paletteName);
        return;
    }
    
    const colorArray = palettePresetsData[paletteName];
    
    // Convert RGB arrays to palette format with hex and proper name (matching main unmix section)
    defaultPaletteData = colorArray.map((rgb, idx) => {
        const r = Math.round(rgb[0]);
        const g = Math.round(rgb[1]);
        const b = Math.round(rgb[2]);
        const hex = rgbToHex(r, g, b);
        return {
            hex: hex,
            name: `${paletteName} #${idx + 1}`
        };
    });
    
    currentTrycolorsPaletteSource = paletteName;
    silencedColors.clear(); // Clear silenced when loading new palette
    renderPaletteGrid();
    
    console.log(`Loaded ${defaultPaletteData.length} colors from "${paletteName}" into Trycolors`);
}

// Event listener for trycolors palette selector
document.getElementById('trycolorsLoadPaletteBtn')?.addEventListener('click', () => {
    const select = document.getElementById('trycolorsPaletteSelect');
    if (select) {
        loadTrycolorsPalette(select.value);
    }
});

// Render palette grid with silenced state
function renderPaletteGrid() {
    let html = '';
    defaultPaletteData.forEach((color, idx) => {
        const isSilenced = silencedColors.has(color.hex);
        html += `<div class="trycolors-palette-chip ${isSilenced ? 'silenced' : ''}" 
            style="background: ${color.hex}" 
            title="${color.name} - ${color.hex}"
            data-hex="${color.hex}"
            data-index="${idx}"
            onclick="setTrycolorsInput('${color.hex}')"
            oncontextmenu="toggleSilencedColor(event, '${color.hex}')"></div>`;
    });
    trycolorsPaletteGrid.innerHTML = html;
    updateSilencedUI();
}

// Toggle silenced state on right-click
function toggleSilencedColor(event, hex) {
    event.preventDefault(); // Prevent context menu
    
    if (silencedColors.has(hex)) {
        silencedColors.delete(hex);
    } else {
        silencedColors.add(hex);
    }
    
    // Update the chip visually
    const chip = event.target.closest('.trycolors-palette-chip');
    if (chip) {
        chip.classList.toggle('silenced', silencedColors.has(hex));
    }
    
    updateSilencedUI();
}

// Update silenced count and controls visibility
function updateSilencedUI() {
    const countEl = document.getElementById('trycolorsSilencedCount');
    const controlsEl = document.getElementById('trycolorsSilencedControls');
    const paletteCountEl = document.getElementById('trycolorsPaletteCount');
    
    const silencedCount = silencedColors.size;
    const activeCount = defaultPaletteData.length - silencedCount;
    
    if (countEl) countEl.textContent = silencedCount;
    if (controlsEl) controlsEl.style.display = silencedCount > 0 ? 'flex' : 'none';
    if (paletteCountEl) paletteCountEl.textContent = activeCount;
}

// Restore all silenced colors
function restoreAllSilencedColors() {
    silencedColors.clear();
    renderPaletteGrid();
}

// Set input from palette click
function setTrycolorsInput(hex) {
    trycolorsInput.value = hex.replace('#', '');
    trycolorsPreview.style.background = hex;
    trycolorsBtn.disabled = false;
    trycolorsInput.focus();
}

// Get recipe button handler - uses same endpoint as main "Get Mix Recipe"
trycolorsBtn.addEventListener('click', async () => {
    let targetColor = trycolorsInput.value.trim().toUpperCase();
    if (!targetColor) return;
    
    // Ensure it has # prefix
    if (!targetColor.startsWith('#')) {
        targetColor = '#' + targetColor;
    }
    
    // Show loading
    trycolorsResults.innerHTML = '<div class="trycolors-loading">Finding best recipe</div>';
    trycolorsBtn.disabled = true;

    // Custom model: run the M7.1-style ranked unmixer over the user's OWN loaded
    // palette (Color Library) via /unmix/custom. Must be checked before the
    // versioned path below, which targets the fixed-palette /version/unmix.
    const selectedUnmixVersion = (document.getElementById('trycolorsModelSelect') || {}).value || 'km_baseline';
    if (selectedUnmixVersion === 'custom') {
        try {
            await runCustomUnmix(targetColor);
        } finally {
            trycolorsBtn.disabled = false;
        }
        return;
    }

    // Versioned models (M7.1 / M7) use the fixed measured palette via a different endpoint.
    if (selectedUnmixVersion && selectedUnmixVersion !== 'km_baseline') {
        try {
            await runVersionedUnmix(targetColor, selectedUnmixVersion);
        } finally {
            trycolorsBtn.disabled = false;
        }
        return;
    }

    try {
        // Get settings from Trycolors controls
        const maxColors = parseInt(document.getElementById('trycolorsMaxColors').value) || 3;
        const maxParts = parseInt(document.getElementById('trycolorsMaxParts').value) || 10;
        const prefilterTopN = parseInt(document.getElementById('prefilterTopNInput').value) || 12;
        const topK = parseInt(document.getElementById('topKInput').value) || 5;
        const gridRows = parseInt(document.getElementById('gridRowsInput').value) || 10;
        const gridCols = parseInt(document.getElementById('gridColsInput').value) || 10;
        const gridMaxDots = parseInt(document.getElementById('gridMaxDotsInput').value) || 0;

        // Build palette from trycolors palette (excluding silenced colors)
        let palette = defaultPaletteData
            .filter(c => !silencedColors.has(c.hex))
            .map(c => ({ hex: c.hex, name: c.name }));

        // If no palette loaded, show error
        if (palette.length === 0) {
            throw new Error('No palette colors available. Please load a palette first.');
        }

        // Use the SAME /unmix endpoint as the main "Get Mix Recipe" button
        const response = await fetch(`${API_BASE}/unmix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_color: targetColor,
                palette: palette,
                max_colors: maxColors,
                max_parts: maxParts,
                prefilter_top_n: prefilterTopN,
                top_k: topK,
                grid_rows: gridRows,
                grid_cols: gridCols,
                grid_max_dots: gridMaxDots
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        renderTrycolorsResultsFromUnmix(data, maxParts);
        
    } catch (error) {
        console.error('Trycolors API error:', error);
        trycolorsResults.innerHTML = `
            <div style="color: #f44336; text-align: center; padding: 20px;">
                ⚠️ API Error: ${error.message}<br>
                <small>Make sure the backend is running at ${API_BASE}</small>
            </div>
        `;
    } finally {
        trycolorsBtn.disabled = false;
    }
});

// Build the shared recipe body (recipe list + dot-matrix/concentric/treemap viz
// + match row + target/result comparison + copyable output). Used by both the
// KM unmix and the versioned (M7/M7.1) renderers so they share one UI.
function buildTrycolorsRecipeHtml(data) {
    let html = '';

    // Recipe items with percentages
    if (data.recipe && data.recipe.length > 0) {
        html += '<div class="trycolors-recipe-list">';
        data.recipe.forEach(item => {
            const percentage = Number(item.percentage).toFixed(1);
            html += `
                <div class="trycolors-recipe-item" style="--clr:${item.hex}; --pct:${percentage}%; border-color: ${item.hex}">
                    <div class="trycolors-recipe-swatch" style="background: ${item.hex}"></div>
                    <div class="trycolors-recipe-info">
                        <div class="trycolors-recipe-name">${item.name || 'Unknown'}</div>
                        <div class="trycolors-recipe-hex">${item.hex}</div>
                    </div>
                    <div class="trycolors-recipe-percent">${percentage}%</div>
                </div>
            `;
        });
        html += '</div>';
        html += buildMixVisualization(data.recipe, parseInt(document.getElementById('gridRowsInput').value) || 10, parseInt(document.getElementById('gridColsInput').value) || 10, parseInt(document.getElementById('gridMaxDotsInput').value) || 0);
    }

    // Match / error row
    const matchPct = Number(data.match_percentage).toFixed(1);
    const errorPct = (100 - Number(data.match_percentage)).toFixed(1);
    html += `
        <div class="trycolors-error-row">
            <div class="trycolors-error-label">Match: ${matchPct}% (ΔE: ${Number(data.delta_e).toFixed(2)})</div>
            <div class="trycolors-error-value">${errorPct}% error</div>
        </div>
    `;

    // Visual comparison
    html += `
        <div class="trycolors-comparison">
            <div>
                <div class="trycolors-comparison-swatch" style="background: ${data.target_color}"></div>
                <div class="trycolors-comparison-label">Target</div>
            </div>
            <div class="trycolors-comparison-arrow">→</div>
            <div>
                <div class="trycolors-comparison-swatch" style="background: ${data.result_color}"></div>
                <div class="trycolors-comparison-label">Result</div>
            </div>
        </div>
    `;

    // Formatted output (copyable)
    let formattedOutput = data.recipe.map(item =>
        `${item.name || 'Color'} ${item.hex} ${Number(item.percentage).toFixed(1)}%`
    ).join('\n');
    formattedOutput += `\nTotal: ${data.total_parts} parts · ΔE: ${Number(data.delta_e).toFixed(2)}`;
    if (data.mix_method) {
        formattedOutput += ` · ${data.mix_method}`;
    }
    html += `
        <div class="trycolors-formatted-output" id="trycolorsOutput">${formattedOutput}</div>
        <button class="trycolors-copy-btn" onclick="copyTrycolorsOutput()">Copy Recipe</button>
    `;

    // Excluded-colors notice (KM path only)
    if (silencedColors.size > 0 && !data._hidesilenced) {
        html += `
            <div style="margin-top: 10px; padding: 8px 12px; background: rgba(255,193,7,0.10); border: 1px solid rgba(255,193,7,0.35); border-radius: 6px; font-size: 11px; color: #ffca28;">
                ⚠️ ${silencedColors.size} color${silencedColors.size > 1 ? 's' : ''} excluded from mixing
            </div>
        `;
    }
    return html;
}

// Render results from /unmix endpoint (same as main recipe) but with trycolors styling
function renderTrycolorsResultsFromUnmix(data, maxParts) {
    _lastTrycolorsData = data;
    _lastTrycolorsMaxParts = maxParts;
    _lastVersionedUnmix = null; // KM result supersedes any versioned result
    if (data.error) {
        trycolorsResults.innerHTML = `<div style="color: #f44336; text-align: center; padding: 20px;">${data.error}</div>`;
        return;
    }
    trycolorsResults.innerHTML = buildTrycolorsRecipeHtml(data);
}

// Allow Enter key to submit
trycolorsInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !trycolorsBtn.disabled) {
        trycolorsBtn.click();
    }
});

/* ==========================================================
   VERSIONED MODELS (M7.1 / M7 / M4 / ...) — single UI, model dropdown
========================================================== */

// The fixed measured 8-pigment palette used by the measured models.
const MEASURED_PIGMENTS = [
    { abbr: 'CY', name: 'Cadmium Yellow Light', hex: '#FEE100' },
    { abbr: 'IY', name: 'India Yellow Hue',     hex: '#AA6200' },
    { abbr: 'CR', name: 'Cadmium Red Light',    hex: '#DE290C' },
    { abbr: 'QM', name: 'Quinacridone Magenta', hex: '#5D162D' },
    { abbr: 'UB', name: 'Ultramarine Blue',     hex: '#19123F' },
    { abbr: 'PG', name: 'Phthalo Green',        hex: '#002E2A' },
    { abbr: 'TW', name: 'Titanium White',       hex: '#F7F5F1' },
    { abbr: 'BK', name: 'Carbon Black',         hex: '#232222' },
];

let _versionRegistry = null;

// Pull version metadata from the backend (notes / availability). Falls back to
// the static <option> lists already in the HTML if the call fails.
async function loadVersionRegistry() {
    try {
        const res = await fetch(`${API_BASE}/versions`);
        if (!res.ok) throw new Error(res.status);
        _versionRegistry = await res.json();
        applyVersionRegistry();
    } catch (e) {
        console.warn('Could not load /versions (using static options):', e.message);
    }
    ensureCustomModelOption();  // idempotent; also covers the static-options fallback
    updateUnmixModelNote();
    updateForwardModelNote();
}

function applyVersionRegistry() {
    if (!_versionRegistry) return;
    const fill = (selectId, items) => {
        const sel = document.getElementById(selectId);
        if (!sel) return;
        const prev = sel.value;
        sel.innerHTML = '';
        items.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.id;
            opt.textContent = v.label + (v.available ? '' : ' (unavailable)');
            if (!v.available) opt.disabled = true;
            sel.appendChild(opt);
        });
        if ([...sel.options].some(o => o.value === prev && !o.disabled)) sel.value = prev;
    };
    fill('trycolorsModelSelect', _versionRegistry.unmix || []);
    fill('fwdModelSelect', _versionRegistry.forward || []);
    // The registry (from /versions) doesn't include the client-only "custom"
    // model, and fill() wipes the <select>, so re-add it afterwards.
    ensureCustomModelOption();
}

// The "Custom (My Colors)" unmix model is handled entirely on the client: it
// runs the M7.1-style unmixer over the palette loaded above (your Color Library)
// via POST /unmix/custom, instead of the fixed measured 8-pigment palette. Make
// sure the option exists even after the registry repopulates the dropdown.
function ensureCustomModelOption() {
    const sel = document.getElementById('trycolorsModelSelect');
    if (!sel) return;
    if ([...sel.options].some(o => o.value === 'custom')) return;
    const opt = document.createElement('option');
    opt.value = 'custom';
    opt.textContent = 'Custom (My Colors)';
    // Place it right after km_baseline if present, else append.
    const km = [...sel.options].find(o => o.value === 'km_baseline');
    if (km && km.nextSibling) sel.insertBefore(opt, km.nextSibling);
    else sel.appendChild(opt);
}

function _versionMeta(mode, id) {
    if (!_versionRegistry) return null;
    return (_versionRegistry[mode] || []).find(v => v.id === id) || null;
}

function updateUnmixModelNote() {
    const sel = document.getElementById('trycolorsModelSelect');
    const note = document.getElementById('trycolorsModelNote');
    if (!sel || !note) return;
    if (sel.value === 'custom') {
        note.textContent = 'Runs the M7.1-style ranked unmixer over the palette loaded above '
            + '(load ★ My Colors to use your saved Color Library). Works with any colors you add.';
        return;
    }
    const v = _versionMeta('unmix', sel.value);
    let txt = v ? (v.note || v.reason || '') : '';
    if (sel.value && sel.value !== 'km_baseline') {
        txt += (txt ? ' ' : '') + 'Uses the fixed measured 8-pigment palette — the loaded palette above does not apply.';
    }
    note.textContent = txt;
}

function updateForwardModelNote() {
    const sel = document.getElementById('fwdModelSelect');
    const note = document.getElementById('fwdModelNote');
    if (!sel || !note) return;
    const v = _versionMeta('forward', sel.value);
    note.textContent = v ? (v.note || v.reason || '') : '';
}

// Convert one measured-model proposal into the shape buildTrycolorsRecipeHtml expects.
function _proposalToRecipeData(data, p) {
    return {
        recipe: p.pigment_abbr.map((ab, k) => ({
            hex: p.pigment_hexes[k],
            name: p.pigment_names[k],
            percentage: p.percentages[k],
        })),
        match_percentage: p.match_percentage,
        delta_e: p.delta_e,
        target_color: data.target_color,
        result_color: p.predicted_hex,
        total_parts: (p.parts || []).reduce((a, b) => a + b, 0),
        mix_method: `${data.version}${p.confidence_tier ? ' · ' + p.confidence_tier : ''}`,
        _hidesilenced: true, // measured models use the fixed palette, not the loaded one
    };
}

// Render proposals from /version/unmix into the SAME UI as the KM unmix
// (dot matrix / concentric / treemap), ranked by highest match on top, with a
// clickable list to inspect each ranked recipe.
function renderVersionedUnmix(data) {
    if (data.error) {
        trycolorsResults.innerHTML = `<div style="color:#f44336; text-align:center; padding:20px;">${data.error}</div>`;
        return;
    }
    if (data.available === false) {
        trycolorsResults.innerHTML = `<div style="background:#fff3cd; border:1px solid #ffc107; color:#856404; padding:14px; border-radius:6px; font-size:12px;">
            <strong>${data.version} is unavailable in this package.</strong><br>${data.reason || ''}</div>`;
        return;
    }
    // Sort: 'match' = highest match % (lowest ΔE) on top; 'reliable' = model's
    // risk-adjusted order (lowest score_with_risk_penalty / model rank first).
    const byMatch = (a, b) => (b.match_percentage - a.match_percentage) || (a.delta_e - b.delta_e);
    const byReliable = (a, b) =>
        ((a.score_with_risk_penalty ?? a.delta_e) - (b.score_with_risk_penalty ?? b.delta_e))
        || ((a.rank ?? 99) - (b.rank ?? 99));
    const proposals = (data.proposals || []).slice()
        .sort(_versionedSortMode === 'reliable' ? byReliable : byMatch);
    if (!proposals.length) {
        trycolorsResults.innerHTML = `<div style="text-align:center; padding:20px; color:#888; font-size:12px;">No proposals returned.</div>`;
        return;
    }

    // Persist for viz-mode re-render and chip selection.
    data._sorted = proposals;
    if (typeof data._selIdx !== 'number' || data._selIdx >= proposals.length) data._selIdx = 0;
    _lastVersionedUnmix = data;
    _lastTrycolorsData = null;

    const sel = data._selIdx;

    // Sort-mode toggle.
    const sortBtn = (mode, label, tip) => {
        const on = _versionedSortMode === mode;
        return `<button type="button" class="versioned-sort-btn${on ? ' active' : ''}" data-sort="${mode}" title="${tip}">${label}</button>`;
    };
    let chips = `<div class="versioned-header">
        <span class="versioned-model-label">Model <strong>${data.version}</strong></span>
        <span class="versioned-sort-group">
            ${sortBtn('match', 'Best match', 'Sort by highest match % (lowest ΔE)')}
            ${sortBtn('reliable', 'Most reliable', "Sort by the model's risk-adjusted score (observed anchors / dense data first)")}
        </span></div>
        <div class="versioned-chips-row">`;
    proposals.forEach((p, i) => {
        const active = i === sel;
        const tierTip = (p.confidence_tier || '') + (p.anchor_trycolors_name ? ' · ' + p.anchor_trycolors_name : '');
        chips += `
            <button type="button" class="versioned-rank-chip${active ? ' active' : ''}" data-idx="${i}" title="${tierTip}">
                <span class="chip-rank">#${i + 1}</span>
                <span class="chip-swatch" style="background:${p.predicted_hex}"></span>
                <span class="chip-match">${Number(p.match_percentage).toFixed(1)}%</span>
                <span class="chip-delta">ΔE ${Number(p.delta_e).toFixed(2)}</span>
            </button>`;
    });
    chips += '</div>';

    // Selected recipe rendered through the shared UI (gets dot-matrix/concentric/treemap).
    const body = buildTrycolorsRecipeHtml(_proposalToRecipeData(data, proposals[sel]));

    // Small confidence line for the selected recipe.
    const selP = proposals[sel];
    const conf = `<div class="versioned-confidence">
        Selected #${sel + 1} · confidence: ${selP.confidence_tier || 'n/a'}${selP.anchor_trycolors_name ? ' · anchor "' + selP.anchor_trycolors_name + '"' : ''} · model risk +${selP.risk_penalty}</div>`;

    trycolorsResults.innerHTML = chips + conf + body;
}

// Chip clicks (event delegation) switch the selected ranked recipe.
document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    // Sort-mode toggle (Best match <-> Most reliable)
    const sortBtn = e.target.closest('.versioned-sort-btn');
    if (sortBtn && _lastVersionedUnmix) {
        _versionedSortMode = sortBtn.dataset.sort === 'reliable' ? 'reliable' : 'match';
        _lastVersionedUnmix._selIdx = 0; // top of the newly-sorted list
        renderVersionedUnmix(_lastVersionedUnmix);
        return;
    }
    // Ranked-recipe chip selection
    const chip = e.target.closest('.versioned-rank-chip');
    if (chip && _lastVersionedUnmix) {
        _lastVersionedUnmix._selIdx = parseInt(chip.dataset.idx) || 0;
        renderVersionedUnmix(_lastVersionedUnmix);
    }
});

async function runVersionedUnmix(targetColor, version) {
    try {
        const maxColors = Math.min(4, parseInt(document.getElementById('trycolorsMaxColors').value) || 4);
        const topK = parseInt(document.getElementById('topKInput').value) || 5;
        const response = await fetch(`${API_BASE}/version/unmix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_color: targetColor,
                version: version,
                max_colors: maxColors,
                total_parts: 6,
                top_n: topK,
            })
        });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        const data = await response.json();
        renderVersionedUnmix(data);
    } catch (error) {
        console.error('Versioned unmix error:', error);
        trycolorsResults.innerHTML = `
            <div style="color:#f44336; text-align:center; padding:20px;">
                ⚠️ API Error: ${error.message}<br>
                <small>Make sure the backend is running at ${API_BASE}</small>
            </div>`;
    }
}

// "Custom (My Colors)" model — runs the M7.1-style ranked unmixer over the
// user's OWN loaded palette (Color Library) via /unmix/custom, and renders the
// result through the same ranked-proposals UI as the measured models.
async function runCustomUnmix(targetColor) {
    // Use the palette currently loaded in the unmixer grid, minus excluded
    // colors. Fall back to the saved Color Library if nothing is loaded.
    let palette = (defaultPaletteData || [])
        .filter(c => !silencedColors.has(c.hex))
        .map(c => ({ hex: c.hex, name: c.name || null }));
    if (palette.length === 0 && window.ColorLibrary && window.ColorLibrary.size() > 0) {
        palette = window.ColorLibrary.asPalette().map(c => ({ hex: c.hex, name: c.name || null }));
    }
    if (palette.length === 0) {
        trycolorsResults.innerHTML = `<div style="background:#fff3cd; border:1px solid #ffc107; color:#856404; padding:14px; border-radius:6px; font-size:12px;">
            <strong>No palette loaded.</strong><br>Load <strong>★ My Colors</strong> (or any palette) above, then try again.</div>`;
        return;
    }
    try {
        const maxColors = parseInt(document.getElementById('trycolorsMaxColors').value) || 4;
        const maxParts = parseInt(document.getElementById('trycolorsMaxParts').value) || 6;
        const prefilterTopN = parseInt(document.getElementById('prefilterTopNInput').value);
        const topK = parseInt(document.getElementById('topKInput').value) || 5;
        const response = await fetch(`${API_BASE}/unmix/custom`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_color: targetColor,
                palette: palette,
                max_colors: Math.min(5, Math.max(1, maxColors)),
                total_parts: Math.max(2, Math.min(12, maxParts)),
                prefilter_top_n: isNaN(prefilterTopN) ? 12 : prefilterTopN,
                top_n: topK,
                mix_method: 'kubelka_munk',
            })
        });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        const data = await response.json();
        renderVersionedUnmix(data);
        // Surface which mixing path ran so the measured profile is visible in the UI.
        try {
            const prof = data.measured_profile;
            let badge;
            if (data.palette_mode === 'custom_measured_profile' && prof) {
                const cov = data.candidate_mix_sources || {};
                badge = `<div style="background:#e6f4ea; border:1px solid #34a853; color:#1e6b32; padding:8px 12px; border-radius:6px; font-size:12px; margin-bottom:10px;">
                    <strong>Measured palette profile in use.</strong> TryColors ${prof.mixer_mode}/${prof.engine}, v${prof.profile_version}
                    (${prof.present_count}/${prof.expected_count} comparisons${prof.complete ? ', complete' : ''}).
                    ${cov.measured_profile || 0} candidates measured, ${cov.physical_km || 0} KM fallback.</div>`;
            } else {
                badge = `<div style="background:#f1f3f4; border:1px solid #bdc1c6; color:#5f6368; padding:8px 12px; border-radius:6px; font-size:12px; margin-bottom:10px;">
                    <strong>Physical KM model.</strong> No measured profile for this palette, so ranking uses the physical mixing model.</div>`;
            }
            trycolorsResults.insertAdjacentHTML('afterbegin', badge);
        } catch (e) { /* badge is best-effort */ }
    } catch (error) {
        console.error('Custom unmix error:', error);
        trycolorsResults.innerHTML = `
            <div style="color:#f44336; text-align:center; padding:20px;">
                ⚠️ API Error: ${error.message}<br>
                <small>Make sure the backend is running at ${API_BASE}</small>
            </div>`;
    }
}


// ---- init (wiring normally done elsewhere in the source app) ----
try { if (typeof loadVersionRegistry === 'function') loadVersionRegistry(); } catch (e) { console.warn('version registry:', e); }
try { if (typeof loadDefaultPalette === 'function') loadDefaultPalette(); } catch (e) { console.warn('default palette:', e); }
try { if (typeof loadPalettePresets === 'function') loadPalettePresets(); } catch (e) { console.warn('palette presets:', e); }
try {
  var _pmSel = document.getElementById('trycolorsModelSelect');
  if (_pmSel && typeof updateUnmixModelNote === 'function') {
    _pmSel.addEventListener('change', updateUnmixModelNote);
    updateUnmixModelNote();
  }
} catch (e) { console.warn('model note:', e); }
