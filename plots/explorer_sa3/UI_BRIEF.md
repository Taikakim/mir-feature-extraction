# SA3 Latent Explorer — UI design brief

*(For a UI redesign pass. Current implementation: Plotly Dash, ~1,900 lines of layout
across the files below. Written 2026-07-08; updated 2026-07-14 — added the sixth tab
(Latent lab), the schedule/interval controls that landed since, and the Deliverables
section.)*

## What this tool is

A single-user, local web app (Dash, `http://localhost:8050`) that started as a music
**dataset viewer** and has grown into a full **instrument for playing a 1.4B-parameter
music diffusion model** (Stable Audio 3 medium). The owner is a musician/producer with
~30 years of experience; the unspoken trajectory is a **DJ / production tool**, not a
lab dashboard. It talks to a model-resident render server over HTTP (FastAPI on
`:8056`) — the UI never touches the model directly.

**Design goal: keep every capability, lose the lab-bench clutter.** The tool
accreted controls feature-by-feature over weeks. Everything works; nothing was ever
*arranged*. The user's actual workflows (below) should each feel like one fluid
surface instead of a wall of numeric inputs.

## The six tabs

### 1. Viewer (track-centric latent browser)
- Dropdown over ~2,676 tracks → 2D scatter of the track's latent embedding
  (coordinates from precomputed projections), plus audio playback of the track.
- Simple today; conceptually the "library" view.

### 2. Dataset (corpus-wide scalar explorer)
- Scatter plot over all crops; X/Y/color selectable from ~40 precomputed scalar
  features (tempo, onset density, spectral stats, LUFS…).
- Point-click → play that crop. Used for finding material and outliers.

### 3. Analysis (correlation views)
- Latent-dim × feature cross-correlation heatmap (256 dims × ~20 features).
- NEW co-activation view: dim-× -feature pointwise-product heatmap over time,
  rows sorted by |r|, sharing a zoom-linked x-axis with a feature timeseries chart.
- Scope selector (400 / 1000 / ALL crops) with scope labels.

### 4. Inference (text → audio; the densest tab, ~85 components)
Sections, top to bottom, all currently plain labeled inputs in rows:
- **Prompt area**: two prompt fields (main + variation), negative prompt, duration,
  batch size, seed (+ randomize), duration-padding.
- **Sampler block**: steps, CFG scale, APG scale, sampler type, **sigma-schedule
  chart** (Plotly line chart of the noise schedule with a draggable CFG-interval
  range — a genuinely novel control, deserves to be a hero element), dist-shift
  (blank = checkpoint default).
- **Checkpoint picker**: recursive-scan dropdown over ~121 checkpoints + strength
  slider + a per-checkpoint "journal" info box (training params, shown on select).
- **Steering panel** (shared component, also used by A2A tab): three collapsible
  groups — LatCH guidance head (head picker, gain, target source), FiLM control
  adapter (ckpt, gain, density target), DoRA adapter selection. 23 interactive
  states total.
- **Weight garden** (databending): shuffle amount / target (attn/mlp/all) / seed /
  decay controls + apply-state readout. Deliberately experimental — could be
  visually quarantined as a "hazard zone".
- **Rhythm-preserve section** (selection steering): enable, K candidates, score
  head, active-fraction slider.
- **Render queue/results**: submit button, hovering progress bar, result list with
  same-playhead audio players (switching clips keeps playhead position).

### 5. A2A Mix (audio-to-audio + DJ transitions; ~60 components)
- Two source-clip pickers (A and B) over an **overlaid waveform canvas**; free
  offsets or BPM-snap of B to A; a draggable, off-centre-capable transition range
  on the waveform.
- Noising controls: strength ladder, sine-schedule (peak/midpoint) toggle,
  seam-inpaint frames, dual prompts, harmonic (chroma) steering default-ON.
- NEW since 07-08 (steering-v2): the sampler's **sigma-interval + distribution-shift
  (schedule) controls now appear here too** — same interaction as the Inference
  sigma chart, so whatever hero treatment that chart gets should be shared.
- Same steering panel + render/results block as Inference.

### 6. Latent lab (latent data-bending; experimental)
- Pick a crop (track→crop browse mirroring the Viewer, or a free `.npy` path),
  stack up to 4 ordered "bend ops" (channel swap/roll, noise, etc. — one seed
  drives all rows, so a recipe reproduces exactly), render via the server,
  A/B the bent result against the crop's clean decode + source.
- Same experimental-toy status as the weight garden: belongs in the visually
  quarantined "hazard zone" treatment, not next to daily controls.

## Hard constraints for the redesign

1. **Keep component IDs.** Every interactive element's Dash `id` is load-bearing —
   ~34 callbacks in `callbacks.py` reference them (a 23-state contract for the
   steering panel alone). A redesign that restyles/re-arranges but preserves IDs
   ships in hours; one that renames IDs costs a callback rewrite.
2. **Dash/Plotly stack** (Python-generated layout, CSS via `assets/`). No React
   rewrite. dcc.Graph charts (sigma schedule, waveforms, heatmaps) stay Plotly.
3. **Same-playhead audio behavior** must survive: result players share playhead
   position when switching clips; re-click stops.
4. **Local, single user, wide desktop screen.** No mobile requirement. Dark theme
   strongly preferred (music-production context; current UI is default-light —
   a known complaint).
5. The render server API is fixed; all controls map 1:1 to request fields.

## Known UX pain points (the actual brief)

- **No visual hierarchy**: one-shot generation (prompt → render) needs 4 controls,
  but they're visually equal to 80 expert knobs. The common path should be
  immediate; the expert surface progressive-disclosure (collapsed groups,
  "advanced" reveals).
- **The sigma chart + CFG interval is the signature interaction** — currently a
  small chart lost mid-form. Make it central to the sampler section.
- **Steering panel** repeats on two tabs with 23 states — wants a compact,
  visually consistent "rack module" treatment (think channel strip / pedalboard).
- **Weight garden + rhythm preserve** are experimental toys sitting next to daily
  controls — separate them visually (color, border, collapsed-by-default).
- **Result players** are an afterthought list; the listening loop (render →
  compare → tweak → re-render) is the core workflow and deserves layout priority
  (e.g. persistent bottom dock).
- **Waveform-with-transition-range** on the A2A tab is the second signature
  interaction; currently cramped. This tab IS the proto-DJ-tool.
- Numeric inputs everywhere that want sliders-with-value or steppers; no reset-to-
  default affordances; no indication which controls are non-default (dirty-state).

## Files that define the UI (send these)

| file | role | lines |
|---|---|---|
| `inference_tab.py` | Inference tab layout (densest surface) | 516 |
| `a2a_tab.py` | A2A Mix tab layout incl. waveform canvas | 637 |
| `controls.py` | Shared steering panel + its state contract | 241 |
| `bend_tab.py` | Latent lab tab (experimental) | 231 |
| `app.py` | Shell: tab container, theme, assets hookup | 46 |
| `audio_panel.py` | Shared audio player panel | 17 |
| `callbacks.py` | (reference only — the ID contract) | 170 |
| `UI_BRIEF.md` | this brief | — |
| `README.md` | architecture/data context (two-process split) | — |

Viewer/Dataset/Analysis tabs (`viewer_tab.py`, `dataset_tab.py`, `analysis_tab.py`)
are simple chart+dropdown layouts and can be restyled by convention once the design
language exists — include them only if the pass has room.

**Note:** there is currently NO `assets/` directory — the app runs on Dash's default
(light) styling. The design pass creates the styling layer from scratch; nothing
existing to fight.

## Deliverables

1. **A design system as `assets/` files** (Dash auto-loads `assets/*.css`):
   dark-theme tokens (palette, typography, spacing, control sizing) in a
   `theme.css`, plus component classes for the recurring patterns — section
   card, collapsible advanced group, "rack module" steering strip, hazard-zone
   (experimental) framing, result-player dock. CSS custom properties so the
   theme is tweakable in one place.
2. **Restyled layout code** for the files above — same Python files, rearranged
   into the hierarchy the pain-points section describes, with `className`s from
   the design system. **Every Dash component `id` preserved verbatim** (the
   acceptance test: `callbacks.py` imports and binds unchanged, zero edits).
3. **A one-page "design language" note** — what the conventions are (when to use
   a card vs a group, hero-element treatment, dirty-state indication), so later
   tabs/features added by other hands stay consistent.
4. *(Optional but valued)* a static mock or annotated screenshot per tab BEFORE
   the code pass, for a quick approval loop.

**Acceptance criteria:** app boots with the same callbacks file untouched; all
six tabs functional; same-playhead behavior intact; dark theme; the 4-control
quick path (prompt → duration → checkpoint → render) is visually primary on the
Inference tab; sigma chart and A2A waveform get hero treatment; experimental
sections visibly quarantined.
