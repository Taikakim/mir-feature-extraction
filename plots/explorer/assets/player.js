/* plots/explorer/assets/player.js
 *
 * Clientside callback: watches the "player-cmd" Store and controls
 * a shared AudioContext + AudioBufferSourceNode.
 *
 * Command schema (stored in player-cmd dcc.Store):
 *   { action: "play"|"pause"|"stop"|"fade",
 *     url:    string,           // absolute URL to WAV stream
 *     slot:   "a"|"b",
 *     loop_start: number|null,  // seconds
 *     loop_end:   number|null,  // seconds
 *     from_hover: bool,
 *     hover_x:    number|null,
 *     hover_y:    number|null,
 *   }
 */
(function () {
    "use strict";

    let _ctx        = null;
    let _sourceNode = null;
    let _gainNode   = null;
    let _fadeTimer  = null;
    let _lastUrl    = null;
    let _hovX       = null;
    let _hovY       = null;

    function getCtx() {
        if (!_ctx) {
            _ctx       = new (window.AudioContext || window.webkitAudioContext)();
            _gainNode  = _ctx.createGain();
            _gainNode.connect(_ctx.destination);
        }
        return _ctx;
    }

    function stopCurrent() {
        if (_sourceNode) {
            try { _sourceNode.stop(); } catch (_) {}
            _sourceNode.disconnect();
            _sourceNode = null;
        }
    }

    function fadeOut(ms) {
        if (!_gainNode) return;
        clearTimeout(_fadeTimer);
        const ctx = getCtx();
        const now = ctx.currentTime;
        _gainNode.gain.cancelScheduledValues(now);
        _gainNode.gain.setValueAtTime(_gainNode.gain.value, now);
        _gainNode.gain.linearRampToValueAtTime(0, now + ms / 1000);
        _fadeTimer = setTimeout(stopCurrent, ms + 50);
    }

    async function playUrl(url, loopStart, loopEnd) {
        if (_lastUrl === url && _sourceNode) return; // already playing
        _lastUrl = url;
        stopCurrent();
        clearTimeout(_fadeTimer);
        const ctx = getCtx();
        await ctx.resume();
        _gainNode.gain.cancelScheduledValues(ctx.currentTime);
        _gainNode.gain.setValueAtTime(1.0, ctx.currentTime);

        let resp;
        try {
            resp = await fetch(url);
            if (!resp.ok) { console.warn("player.js: fetch failed", resp.status); return; }
        } catch (e) { console.warn("player.js: fetch error", e); return; }

        const buf = await resp.arrayBuffer();
        let decoded;
        try { decoded = await ctx.decodeAudioData(buf); }
        catch (e) { console.warn("player.js: decode error", e); return; }

        _sourceNode = ctx.createBufferSource();
        _sourceNode.buffer = decoded;
        _sourceNode.connect(_gainNode);

        if (loopStart != null && loopEnd != null && loopEnd > loopStart) {
            _sourceNode.loop      = true;
            _sourceNode.loopStart = loopStart;
            _sourceNode.loopEnd   = loopEnd;
            _sourceNode.start(0, loopStart);
        } else {
            _sourceNode.loop = false;
            _sourceNode.start(0);
        }
    }

    // ── Hover proximity fade ─────────────────────────────────────────────────
    document.addEventListener("mousemove", function (e) {
        if (_hovX === null) return;
        const dx = e.clientX - _hovX;
        const dy = e.clientY - _hovY;
        if (dx * dx + dy * dy > 100) { // > 10px movement
            fadeOut(200);
            _hovX = null; _hovY = null;
        }
    });

    function setHoverOrigin(x, y) {
        _hovX = x; _hovY = y;
    }

    // ── Dash clientside callback registration ────────────────────────────────
    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.player = {
        handle_cmd: function (cmd) {
            if (!cmd || !cmd.action) return window.dash_clientside.no_update;
            if (cmd.action === "play") {
                playUrl(cmd.url, cmd.loop_start ?? null, cmd.loop_end ?? null);
                if (cmd.from_hover) setHoverOrigin(cmd.hover_x ?? null, cmd.hover_y ?? null);
            } else if (cmd.action === "stop") {
                stopCurrent(); _lastUrl = null;
            } else if (cmd.action === "fade") {
                fadeOut(cmd.ms ?? 200);
            } else if (cmd.action === "pause") {
                if (_ctx) _ctx.suspend();
            }
            return window.dash_clientside.no_update;
        },
    };
}());
