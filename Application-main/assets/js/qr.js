/* Minimal QR code encoder (byte mode, error-correction level M, versions 1-10).
 * Enough for the device pairing deep link on the Devices page; no dependencies.
 *
 *   var m = QR.encode("alchroma://pair?code=ABCD2345");   // m.size, m.get(x, y)
 *   QR.draw(canvas, "text", { scale: 6, margin: 4 });     // paints onto a <canvas>
 */
(function () {
   // ---- GF(256) arithmetic (prime polynomial 0x11D) ---------------------
   var EXP = new Array(512), LOG = new Array(256);
   (function () {
      var x = 1;
      for (var i = 0; i < 255; i++) { EXP[i] = x; LOG[x] = i; x <<= 1; if (x & 0x100) x ^= 0x11D; }
      for (var j = 255; j < 512; j++) EXP[j] = EXP[j - 255];
   })();
   function mul(a, b) { return (a === 0 || b === 0) ? 0 : EXP[LOG[a] + LOG[b]]; }

   function rsGenerator(n) {
      var g = [1];
      for (var i = 0; i < n; i++) {
         var next = new Array(g.length + 1).fill(0);
         for (var j = 0; j < g.length; j++) {
            next[j] ^= g[j];
            next[j + 1] ^= mul(g[j], EXP[i]);
         }
         g = next;
      }
      return g;
   }
   function rsEncode(data, n) {
      var gen = rsGenerator(n), res = new Array(n).fill(0);
      for (var i = 0; i < data.length; i++) {
         var f = data[i] ^ res[0];
         res.shift(); res.push(0);
         if (f !== 0) for (var j = 0; j < n; j++) res[j] ^= mul(gen[j + 1], f);
      }
      return res;
   }

   // ---- Version tables (level M): [total codewords, ec per block, [ [count, dataLen], ... ]] ----
   var VERSIONS = {
      1:  [26,  10, [[1, 16]]],
      2:  [44,  16, [[1, 28]]],
      3:  [70,  26, [[1, 44]]],
      4:  [100, 18, [[2, 32]]],
      5:  [134, 24, [[2, 43]]],
      6:  [172, 16, [[4, 27]]],
      7:  [196, 18, [[4, 31]]],
      8:  [242, 22, [[2, 38], [2, 39]]],
      9:  [292, 22, [[3, 36], [2, 37]]],
      10: [346, 26, [[4, 43], [1, 44]]]
   };
   var ALIGN = { 1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50] };

   function dataCapacity(v) {
      var n = 0; VERSIONS[v][2].forEach(function (b) { n += b[0] * b[1]; }); return n;
   }
   function utf8(str) {
      var out = [], s = unescape(encodeURIComponent(str));
      for (var i = 0; i < s.length; i++) out.push(s.charCodeAt(i));
      return out;
   }

   function buildCodewords(bytes, v) {
      var cap = dataCapacity(v), bits = [];
      function put(val, len) { for (var i = len - 1; i >= 0; i--) bits.push((val >>> i) & 1); }
      put(4, 4);                                   // byte mode
      put(bytes.length, v < 10 ? 8 : 16);          // char count
      bytes.forEach(function (b) { put(b, 8); });
      var max = cap * 8;
      put(0, Math.min(4, max - bits.length));      // terminator
      while (bits.length % 8) bits.push(0);
      var data = [];
      for (var i = 0; i < bits.length; i += 8) {
         var b = 0; for (var j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
         data.push(b);
      }
      for (var p = 0; data.length < cap; p++) data.push(p % 2 ? 0x11 : 0xEC);

      // split into blocks, compute EC, interleave
      var spec = VERSIONS[v], ecLen = spec[1], blocks = [], ecs = [], pos = 0;
      spec[2].forEach(function (grp) {
         for (var k = 0; k < grp[0]; k++) {
            var blk = data.slice(pos, pos + grp[1]); pos += grp[1];
            blocks.push(blk); ecs.push(rsEncode(blk, ecLen));
         }
      });
      var out = [], maxLen = 0;
      blocks.forEach(function (b) { maxLen = Math.max(maxLen, b.length); });
      for (var c = 0; c < maxLen; c++) blocks.forEach(function (b) { if (c < b.length) out.push(b[c]); });
      for (var e = 0; e < ecLen; e++) ecs.forEach(function (b) { out.push(b[e]); });
      return out;
   }

   // ---- Matrix ---------------------------------------------------------
   function Matrix(v) {
      this.v = v; this.size = v * 4 + 17;
      this.m = []; this.fn = [];             // module values and "is function pattern" flags
      for (var i = 0; i < this.size; i++) { this.m.push(new Array(this.size).fill(0)); this.fn.push(new Array(this.size).fill(false)); }
   }
   Matrix.prototype.set = function (x, y, val, isFn) { this.m[y][x] = val ? 1 : 0; if (isFn) this.fn[y][x] = true; };
   Matrix.prototype.get = function (x, y) { return this.m[y][x]; };

   function placeFinder(mx, cx, cy) {
      for (var dy = -4; dy <= 4; dy++) for (var dx = -4; dx <= 4; dx++) {
         var x = cx + dx, y = cy + dy;
         if (x < 0 || y < 0 || x >= mx.size || y >= mx.size) continue;
         var d = Math.max(Math.abs(dx), Math.abs(dy));
         mx.set(x, y, d !== 2 && d !== 4, true);
      }
   }
   function placeAlign(mx, cx, cy) {
      for (var dy = -2; dy <= 2; dy++) for (var dx = -2; dx <= 2; dx++) {
         var d = Math.max(Math.abs(dx), Math.abs(dy));
         mx.set(cx + dx, cy + dy, d !== 1, true);
      }
   }
   function placeFunctionPatterns(mx) {
      var n = mx.size;
      placeFinder(mx, 3, 3); placeFinder(mx, n - 4, 3); placeFinder(mx, 3, n - 4);
      for (var i = 8; i < n - 8; i++) { mx.set(i, 6, i % 2 === 0, true); mx.set(6, i, i % 2 === 0, true); }
      var al = ALIGN[mx.v];
      for (var a = 0; a < al.length; a++) for (var b = 0; b < al.length; b++) {
         var x = al[a], y = al[b];
         if ((x === 6 && y === 6) || (x === 6 && y === n - 7) || (x === n - 7 && y === 6)) continue;
         placeAlign(mx, x, y);
      }
      // reserve format areas (values written later) + dark module; (6,8)/(8,6) are timing modules, keep them
      for (var k = 0; k < 9; k++) { if (!mx.fn[8][k]) mx.set(k, 8, 0, true); if (!mx.fn[k][8]) mx.set(8, k, 0, true); }
      for (var q = 0; q < 8; q++) { mx.set(n - 1 - q, 8, 0, true); mx.set(8, n - 1 - q, 0, true); }
      mx.set(8, n - 8, 1, true);
      if (mx.v >= 7) for (var t = 0; t < 6; t++) for (var u = 0; u < 3; u++) { mx.set(t, n - 11 + u, 0, true); mx.set(n - 11 + u, t, 0, true); }
   }
   function placeData(mx, codewords) {
      var n = mx.size, bitIdx = 0, total = codewords.length * 8;
      function bit(i) { return (codewords[i >> 3] >>> (7 - (i & 7))) & 1; }
      for (var right = n - 1; right >= 1; right -= 2) {
         if (right === 6) right = 5;
         for (var vert = 0; vert < n; vert++) {
            for (var j = 0; j < 2; j++) {
               var x = right - j, upward = ((right + 1) & 2) === 0, y = upward ? n - 1 - vert : vert;
               if (mx.fn[y][x]) continue;
               mx.m[y][x] = bitIdx < total ? bit(bitIdx) : 0;
               bitIdx++;
            }
         }
      }
   }
   function maskBit(mask, x, y) {
      switch (mask) {
         case 0: return (x + y) % 2 === 0;
         case 1: return y % 2 === 0;
         case 2: return x % 3 === 0;
         case 3: return (x + y) % 3 === 0;
         case 4: return (Math.floor(y / 2) + Math.floor(x / 3)) % 2 === 0;
         case 5: return (x * y) % 2 + (x * y) % 3 === 0;
         case 6: return ((x * y) % 2 + (x * y) % 3) % 2 === 0;
         default: return ((x + y) % 2 + (x * y) % 3) % 2 === 0;
      }
   }
   function applyMask(mx, mask) {
      for (var y = 0; y < mx.size; y++) for (var x = 0; x < mx.size; x++) if (!mx.fn[y][x] && maskBit(mask, x, y)) mx.m[y][x] ^= 1;
   }
   // value * x^degree(poly) plus the remainder of dividing by poly (BCH code word).
   function bch(value, poly) {
      var degree = 0; while ((poly >> degree) > 1) degree++;
      var v = value << degree;
      for (var i = 30; i >= degree; i--) if (v & (1 << i)) v ^= poly << (i - degree);
      return (value << degree) | v;
   }
   function writeFormat(mx, mask) {
      var n = mx.size, data = (0 << 3) | mask;                 // level M = 00
      var f = bch(data, 0x537) ^ 0x5412;
      function gb(i) { return (f >> i) & 1; }                  // bit i, LSB = 0 (ISO 18004 ordering)
      // copy 1: around the top-left finder
      for (var i = 0; i <= 5; i++) mx.set(8, i, gb(i), true);
      mx.set(8, 7, gb(6), true); mx.set(8, 8, gb(7), true); mx.set(7, 8, gb(8), true);
      for (var j = 9; j < 15; j++) mx.set(14 - j, 8, gb(j), true);
      // copy 2: top-right row and bottom-left column
      for (var a = 0; a < 8; a++) mx.set(n - 1 - a, 8, gb(a), true);
      for (var b = 8; b < 15; b++) mx.set(8, n - 15 + b, gb(b), true);
      mx.set(8, n - 8, 1, true);
      if (mx.v >= 7) {
         var vi = bch(mx.v, 0x1F25);
         for (var t = 0; t < 18; t++) {
            var bitv = (vi >> t) & 1, p = n - 11 + (t % 3), q = Math.floor(t / 3);
            mx.set(p, q, bitv, true); mx.set(q, p, bitv, true);
         }
      }
   }
   function penalty(mx) {
      var n = mx.size, score = 0, m = mx.m, x, y;
      function runs(get) {
         for (var i = 0; i < n; i++) {
            var run = 1;
            for (var j = 1; j < n; j++) {
               if (get(i, j) === get(i, j - 1)) { run++; if (run === 5) score += 3; else if (run > 5) score++; }
               else run = 1;
            }
         }
      }
      runs(function (i, j) { return m[i][j]; });
      runs(function (i, j) { return m[j][i]; });
      for (y = 0; y < n - 1; y++) for (x = 0; x < n - 1; x++) {
         var s = m[y][x] + m[y][x + 1] + m[y + 1][x] + m[y + 1][x + 1];
         if (s === 0 || s === 4) score += 3;
      }
      var pat = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], rev = pat.slice().reverse();
      function finderLike(get) {
         for (var i = 0; i < n; i++) for (var j = 0; j <= n - 11; j++) {
            var ok1 = true, ok2 = true;
            for (var k = 0; k < 11; k++) { var v = get(i, j + k); if (v !== pat[k]) ok1 = false; if (v !== rev[k]) ok2 = false; }
            if (ok1 || ok2) score += 40;
         }
      }
      finderLike(function (i, j) { return m[i][j]; });
      finderLike(function (i, j) { return m[j][i]; });
      var dark = 0; for (y = 0; y < n; y++) for (x = 0; x < n; x++) dark += m[y][x];
      score += Math.floor(Math.abs(dark * 100 / (n * n) - 50) / 5) * 10;
      return score;
   }

   function encode(text) {
      var bytes = utf8(text), v = 1;
      while (v <= 10 && dataCapacity(v) < bytes.length + 2 + (v >= 10 ? 1 : 0)) v++;
      if (v > 10) throw new Error('QR: text too long (max ~200 bytes)');
      var codewords = buildCodewords(bytes, v);
      var best = null, bestScore = Infinity;
      for (var mask = 0; mask < 8; mask++) {
         var mx = new Matrix(v);
         placeFunctionPatterns(mx);
         placeData(mx, codewords);
         applyMask(mx, mask);
         writeFormat(mx, mask);
         var sc = penalty(mx);
         if (sc < bestScore) { bestScore = sc; best = mx; best.mask = mask; }
      }
      return { size: best.size, version: v, mask: best.mask, get: function (x, y) { return best.m[y][x]; }, rows: best.m };
   }

   function draw(canvas, text, opts) {
      opts = opts || {};
      var q = encode(text), scale = opts.scale || 6, margin = opts.margin == null ? 4 : opts.margin;
      var px = (q.size + margin * 2) * scale;
      canvas.width = px; canvas.height = px;
      var ctx = canvas.getContext('2d');
      ctx.fillStyle = opts.light || '#ffffff'; ctx.fillRect(0, 0, px, px);
      ctx.fillStyle = opts.dark || '#111315';
      for (var y = 0; y < q.size; y++) for (var x = 0; x < q.size; x++) if (q.get(x, y)) ctx.fillRect((x + margin) * scale, (y + margin) * scale, scale, scale);
      return q;
   }

   window.QR = { encode: encode, draw: draw };
   if (typeof module !== 'undefined' && module.exports) module.exports = window.QR;
})();
