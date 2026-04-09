/* plots/explorer/assets/player.js
 *
 * Command schema (stored in player-cmd dcc.Store):
 *   { action: "play"|"stop"|"fade"|"prefetch",
 *     url:           string,
 *     loop_start:    number|null,
 *     loop_end:      number|null,
 *     from_hover:    bool,
 *     hover_x:       number|null,
 *     hover_y:       number|null,
 *     continue_auto: bool,     // advance to next crop when current ends
 *     vae_fade:      bool,     // VAE-blend crossfade into next crop
 *   }
 */
(function () {
    "use strict";

    // ── AudioContext & master gain ────────────────────────────────────────────
    let _ctx        = null;
    let _gainNode   = null;   // master gain → destination
    let _fadeTimer  = null;

    // ── Current playback ──────────────────────────────────────────────────────
    let _sourceNode = null;
    let _lastUrl    = null;
    let _loopStart  = null;
    let _loopEnd    = null;
    let _playGen    = 0;   // incremented on every new play request; guards async races

    // ── Crop tracking (read from response headers) ────────────────────────────
    let _cropIndex  = null;   // 1-based index of currently-playing crop
    let _cropCount  = null;   // total crops in this track
    let _cropPos    = null;   // actual position value of current crop

    // ── Prefetch / next-crop buffer ───────────────────────────────────────────
    let _pending = null;  // { url, buffer, cropIndex, cropCount, cropPos }

    // ── Crossfade (VAE fade) blend buffer ─────────────────────────────────────
    let _blend = null;  // { url, buffer }

    // ── Queued next track (set by fade_to when crossfade can't start yet) ────
    let _pendingNext = null;  // { url, loopStart, loopEnd }

    // ── Options ───────────────────────────────────────────────────────────────
    let _continueAuto = false;
    let _vaeFade      = false;

    // ── Hover ─────────────────────────────────────────────────────────────────
    let _hovX = null;
    let _hovY = null;

    // ── Helpers ───────────────────────────────────────────────────────────────
    function getCtx() {
        if (!_ctx) {
            _ctx      = new (window.AudioContext || window.webkitAudioContext)();
            _gainNode = _ctx.createGain();
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

    /** Compute the URL for crop index (1-based) `idx` given a base decode URL. */
    function _urlAtCropIdx(baseUrl, idx, count) {
        if (!baseUrl || !idx || !count || idx > count) return null;
        try {
            const u = new URL(baseUrl);
            u.searchParams.set("position", (idx / count).toFixed(4));
            return u.toString();
        } catch (_) { return null; }
    }

    /**
     * Build a /crossfade URL blending srcUrl@posA → destUrl@posB at mix=0.5.
     * destUrl defaults to srcUrl (same-track blend when omitted).
     */
    function _blendUrl(srcUrl, posA, posB, destUrl) {
        if (!srcUrl || posA == null || posB == null) return null;
        try {
            const s    = new URL(srcUrl);
            const d    = destUrl ? new URL(destUrl) : s;
            const trackA = s.searchParams.get("track");
            const trackB = d.searchParams.get("track") || trackA;
            const interp = s.searchParams.get("interp") || "slerp";
            const sl     = s.searchParams.get("smart_loop") || "0";
            if (!trackA) return null;

            const xf = new URL(s.origin + "/crossfade");
            xf.searchParams.set("track_a",    trackA);
            xf.searchParams.set("position_a", posA.toFixed(4));
            xf.searchParams.set("track_b",    trackB);
            xf.searchParams.set("position_b", posB.toFixed(4));
            xf.searchParams.set("mix",        "0.500");
            xf.searchParams.set("interp",     interp);
            if (sl === "1") xf.searchParams.set("smart_loop", "1");
            return xf.toString();
        } catch (_) { return null; }
    }

    /** Extract position param from a decode URL (or null). */
    function _urlPos(url) {
        try { const v = parseFloat(new URL(url).searchParams.get("position")); return isNaN(v) ? null : v; }
        catch (_) { return null; }
    }

    /** Safe header parser — returns number or null (avoids 0 || null bug). */
    function _hdr(resp, name) {
        const v = parseFloat(resp.headers.get(name));
        return isNaN(v) ? null : v;
    }
    function _hdrInt(resp, name) {
        const v = parseInt(resp.headers.get(name));
        return isNaN(v) ? null : v;
    }

    /** Fetch a URL, return {buffer, cropIndex, cropCount, cropPos} or null on error. */
    async function _fetchDecode(url) {
        const ctx = getCtx();
        let resp;
        try {
            resp = await fetch(url);
            if (!resp.ok) { console.warn("player.js: fetch failed", resp.status, url); return null; }
        } catch (e) { console.warn("player.js: fetch error", e); return null; }

        const cropIndex = _hdrInt(resp, "X-Crop-Index");
        const cropCount = _hdrInt(resp, "X-Crop-Count");
        const cropPos   = _hdr(resp,    "X-Crop-Position");

        const raw = await resp.arrayBuffer();
        let buffer;
        try { buffer = await ctx.decodeAudioData(raw); }
        catch (e) { console.warn("player.js: decode error", e); return null; }

        return { buffer, cropIndex, cropCount, cropPos };
    }

    /** Start a decoded buffer. Installs onended for continue-auto logic. */
    function _startBuffer(decoded, loopStart, loopEnd) {
        const ctx = getCtx();
        _gainNode.gain.cancelScheduledValues(ctx.currentTime);
        _gainNode.gain.setValueAtTime(1.0, ctx.currentTime);

        const src = ctx.createBufferSource();
        src.buffer = decoded.buffer;
        src.connect(_gainNode);

        _sourceNode  = src;
        _cropIndex   = decoded.cropIndex;
        _cropCount   = decoded.cropCount;
        _cropPos     = decoded.cropPos;

        // If smart loop (sentinel 0,-1): normally loop the buffer.
        // But if continue_auto, play once and advance on ended instead of looping.
        if (loopStart === 0 && loopEnd === -1) {
            if (_continueAuto) {
                src.loop = false;
            } else {
                src.loop      = true;
                src.loopStart = 0;
                src.loopEnd   = decoded.buffer.duration;
            }
            src.start(0, 0);
        } else if (loopStart != null && loopEnd != null && loopEnd > loopStart) {
            if (_continueAuto) {
                src.loop = false;
            } else {
                src.loop      = true;
                src.loopStart = loopStart;
                src.loopEnd   = loopEnd;
            }
            src.start(0, loopStart);
        } else {
            src.loop = false;
            src.start(0);
        }

        src.onended = function () {
            if (_continueAuto && src === _sourceNode) {
                _onClipEnded();
            }
        };
    }

    /** Called when the current clip finishes and continue_auto is on. */
    async function _onClipEnded() {
        // ── Check if user queued a new track ─────────────────────────────────
        if (_pendingNext) {
            const pn     = _pendingNext;
            _pendingNext = null;
            _loopStart   = pn.loopStart;
            _loopEnd     = pn.loopEnd;
            const posA   = _cropPos ?? (_cropIndex != null && _cropCount
                            ? _cropIndex / _cropCount : null);
            const posB   = _urlPos(pn.url);
            if (_vaeFade && posA != null) {
                const bUrl = _blendUrl(_lastUrl, posA, posB ?? 0.5, pn.url);
                if (bUrl) {
                    _lastUrl = pn.url;
                    await _playWithFade(bUrl, pn.url, pn.loopStart, pn.loopEnd);
                    return;
                }
            }
            await _playNext(pn.url, null);
            return;
        }

        // ── Normal auto-advance within current track ──────────────────────────
        const nextIdx = _cropIndex != null ? _cropIndex + 1 : null;
        const nextUrl = _urlAtCropIdx(_lastUrl, nextIdx, _cropCount);
        if (!nextUrl) return;  // last crop reached

        if (_vaeFade) {
            const posA = _cropPos ?? (_cropIndex != null && _cropCount
                          ? _cropIndex / _cropCount : null);
            const posB = nextIdx != null && _cropCount ? nextIdx / _cropCount : null;
            const bUrl = _blendUrl(_lastUrl, posA, posB);

            let blendDecoded = null;
            if (_blend && _blend.url === bUrl) {
                blendDecoded = { buffer: _blend.buffer, cropIndex: _cropIndex,
                                 cropCount: _cropCount, cropPos: _cropPos };
            } else if (bUrl) {
                blendDecoded = await _fetchDecode(bUrl);
            }

            if (blendDecoded) {
                stopCurrent();
                _startBuffer(blendDecoded, null, null);
                const blendSrc = _sourceNode;
                blendSrc.onended = async function () {
                    if (blendSrc !== _sourceNode) return;
                    await _playNext(nextUrl, nextIdx);
                };
                _prefetchUrl(nextUrl, nextIdx);
                return;
            }
        }

        await _playNext(nextUrl, nextIdx);
    }

    /** Play the next crop (used pending buffer if URL matches). */
    async function _playNext(nextUrl, nextIdx) {
        const gen = ++_playGen;
        stopCurrent();
        _lastUrl = nextUrl;

        let decoded;
        if (_pending && _pending.url === nextUrl) {
            decoded  = _pending;
            _pending = null;
        } else {
            decoded = await _fetchDecode(nextUrl);
            if (!decoded) return;
        }

        if (gen !== _playGen) return;  // superseded
        _startBuffer(decoded, _loopStart, _loopEnd);

        // Pre-decode the one after next
        const afterIdx = decoded.cropIndex != null ? decoded.cropIndex + 1 : (nextIdx || 0) + 1;
        _prefetchUrl(_urlAtCropIdx(nextUrl, afterIdx, decoded.cropCount || _cropCount), afterIdx);
        if (_vaeFade) {
            const nPos = decoded.cropPos || (afterIdx > 1 ? (afterIdx - 1) / decoded.cropCount : 0.5);
            const aPos = (afterIdx) / (decoded.cropCount || _cropCount);
            _prefetchBlend(_lastUrl, nPos, aPos);
        }
    }

    /** Background prefetch a decode URL into _pending. */
    async function _prefetchUrl(url, expectedIdx) {
        if (!url || (_pending && _pending.url === url)) return;
        _pending = null;  // clear stale pending
        const decoded = await _fetchDecode(url);
        if (!decoded) return;
        // Only store if nothing else took over meanwhile
        _pending = decoded;
        _pending.url = url;
    }

    /** Background prefetch a VAE blend clip into _blend. */
    async function _prefetchBlend(baseUrl, posA, posB) {
        const url = _blendUrl(baseUrl, posA, posB);
        if (!url || (_blend && _blend.url === url)) return;
        _blend = null;
        const decoded = await _fetchDecode(url);
        if (!decoded) return;
        _blend = { url, buffer: decoded.buffer };
    }

    // ── Main playUrl (called by play command) ─────────────────────────────────
    async function playUrl(url, loopStart, loopEnd) {
        if (_lastUrl === url && _sourceNode) return;  // already playing this
        const gen  = ++_playGen;  // capture; any later call will bump this higher
        _lastUrl   = url;
        _loopStart = loopStart;
        _loopEnd   = loopEnd;
        stopCurrent();
        clearTimeout(_fadeTimer);

        const ctx = getCtx();
        await ctx.resume();

        let decoded;
        // Use pending buffer if it matches the requested URL
        if (_pending && _pending.url === url) {
            decoded  = _pending;
            _pending = null;
        } else {
            decoded = await _fetchDecode(url);
            if (!decoded) return;
        }

        if (gen !== _playGen) return;  // superseded by a newer play request
        _startBuffer(decoded, loopStart, loopEnd);

        // If continue_auto, immediately prefetch the next crop
        if (_continueAuto && decoded.cropIndex && decoded.cropCount) {
            const nextIdx = decoded.cropIndex + 1;
            const nextUrl = _urlAtCropIdx(url, nextIdx, decoded.cropCount);
            _prefetchUrl(nextUrl, nextIdx);
            if (_vaeFade && decoded.cropPos != null) {
                const posA = decoded.cropPos;
                const posB = nextIdx / decoded.cropCount;
                _prefetchBlend(url, posA, posB);
            }
        }
    }

    // ── Hover proximity fade ──────────────────────────────────────────────────
    document.addEventListener("mousemove", function (e) {
        if (_hovX === null) return;
        const dx = e.clientX - _hovX;
        const dy = e.clientY - _hovY;
        if (dx * dx + dy * dy > 100) {
            fadeOut(200);
            _hovX = null; _hovY = null;
        }
    });

    // ── Shared helper: play blend clip then loop destination ─────────────────
    async function _playWithFade(blendUrl, destUrl, loopStart, loopEnd) {
        const gen          = ++_playGen;
        const blendDecoded = await _fetchDecode(blendUrl);
        if (gen !== _playGen) return;  // superseded
        stopCurrent();
        _lastUrl   = destUrl;
        _loopStart = loopStart;
        _loopEnd   = loopEnd;
        if (blendDecoded) {
            _startBuffer(blendDecoded, null, null);
            const blendSrc = _sourceNode;
            _prefetchUrl(destUrl, null);
            blendSrc.onended = async function () {
                if (blendSrc !== _sourceNode || gen !== _playGen) return;
                await _playNext(destUrl, null);
            };
        } else {
            await _playNext(destUrl, null);
        }
    }

    // ── Dash clientside callback entry point ──────────────────────────────────
    window.dash_clientside = window.dash_clientside || {};
    window.dash_clientside.player = {
        handle_cmd: function (cmd) {
            if (!cmd || !cmd.action) return window.dash_clientside.no_update;

            if (cmd.action === "play") {
                // Sync option flags
                _continueAuto = !!cmd.continue_auto;
                _vaeFade      = !!cmd.vae_fade;
                playUrl(cmd.url, cmd.loop_start ?? null, cmd.loop_end ?? null);
                if (cmd.from_hover) { _hovX = cmd.hover_x ?? null; _hovY = cmd.hover_y ?? null; }

            } else if (cmd.action === "play_with_fade") {
                // Decode blend → play blend → then loop dest
                _continueAuto = !!cmd.continue_auto;
                _vaeFade      = !!cmd.vae_fade;
                _loopStart    = cmd.loop_start ?? null;
                _loopEnd      = cmd.loop_end   ?? null;
                _playWithFade(cmd.blend_url, cmd.dest_url, _loopStart, _loopEnd);

            } else if (cmd.action === "fade_to") {
                // Transition to a new track — immediate crossfade if possible,
                // queue to _pendingNext if not (never switches abruptly mid-crop).
                const destUrl  = cmd.url;
                const loopSt   = cmd.loop_start ?? null;
                const loopEd   = cmd.loop_end   ?? null;
                if (!destUrl) return window.dash_clientside.no_update;

                if (!_sourceNode || !_lastUrl) {
                    // Nothing is playing — start the new track immediately
                    playUrl(destUrl, loopSt, loopEd);
                    return window.dash_clientside.no_update;
                }

                // Something is playing — try an immediate crossfade
                const posA = _cropPos ?? (_cropIndex != null && _cropCount
                              ? _cropIndex / _cropCount : null);
                const posB = _urlPos(destUrl);

                if (_vaeFade && posA != null) {
                    const bUrl = _blendUrl(_lastUrl, posA, posB ?? 0.5, destUrl);
                    if (bUrl) {
                        _loopStart = loopSt; _loopEnd = loopEd;
                        _playWithFade(bUrl, destUrl, loopSt, loopEd);
                        return window.dash_clientside.no_update;
                    }
                }

                // Can't crossfade (position unknown, or vaeFade off) —
                // queue for the next crop boundary instead of interrupting.
                _pendingNext = { url: destUrl, loopStart: loopSt, loopEnd: loopEd };

            } else if (cmd.action === "options") {
                // Sync flags without restarting playback
                _continueAuto = !!cmd.continue_auto;
                _vaeFade      = !!cmd.vae_fade;

            } else if (cmd.action === "prefetch") {
                // Background decode on slider release — use pending slot
                const url = cmd.url;
                if (!url || (_pending && _pending.url === url)) return window.dash_clientside.no_update;
                _pending = null;
                _prefetchUrl(url, null);

            } else if (cmd.action === "stop") {
                ++_playGen;  // invalidate any in-flight fetches
                stopCurrent();
                _lastUrl     = null;
                _pending     = null;
                _blend       = null;
                _pendingNext = null;

            } else if (cmd.action === "fade") {
                fadeOut(cmd.ms ?? 200);
            } else if (cmd.action === "pause") {
                if (_ctx) _ctx.suspend();
            }

            return window.dash_clientside.no_update;
        },
    };
}());
