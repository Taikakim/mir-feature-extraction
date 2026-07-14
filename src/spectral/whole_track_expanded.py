"""
Expanded whole-track descriptors — Kim's expanded-Essentia sweep (2026-07-14).

Companion to whole_track_timeseries.py: adds model-based and plain-DSP
time-series fields to the .TIMESERIES.npz sidecars. Field list + design calls
confirmed by CONTINUITY on the fleet channel 2026-07-14 (validation gates:
MAEST washout NN-retrieval, stereo-width broken-render separation, add-fields
equivalence — run on a ~20-track pilot before any full-corpus sweep).

Unlike the legacy fields (all fitted to n_frames at 100 Hz), expanded fields
land at their own NATIVE rates — embeddings are coarse by design (they serve
rarity + clip metrics, not frame-accurate curves). The sidecar meta gains
`field_rates` {field: fps}; consumers MUST use it for any field not at the
legacy `frame_rate`.

Model-based (essentia-tensorflow; the TF build is CPU-only and
CUDA_VISIBLE_DEVICES is cleared before import — this pins TF's CUDA probe,
NOT HIP, so it never trips the flash-attn/aiter import crash):
  maest_embed_ts        (n, 768)  ~0.2 Hz  MAEST discogs-maest-10s-fs-2, layer-7
                                           token-mean (the metadata-recommended
                                           embedding layer), patch hop 5 s
  (OpenL3 was DROPPED by C's gate-(a) verdict 2026-07-14: top-1 washout
   retrieval 44.6% vs the 48.6% raw-mel baseline — measurably worse than free
   at content retrieval across the production gap, so it does not enter the
   frozen field set. The working extractor survives in git history; a second
   embedding for triangulation (MERT/CLAP) is a pool item, not a freeze blocker.)
  effnet_genre400_ts    (n, 400)  ~1 Hz    discogs-effnet-bs64 -> genre_discogs400
  effnet_moodtheme_ts   (n, 56)   ~1 Hz    -> mtg_jamendo_moodtheme (sigmoid)
  effnet_instrument_ts  (n, 40)   ~1 Hz    -> mtg_jamendo_instrument (sigmoid)
  va_deam_ts            (n, 2)    ~1.04 Hz VGGish -> DEAM (valence, arousal), 1-9
  va_emomusic_ts        (n, 2)    ~1.04 Hz VGGish -> emoMusic (valence, arousal), 1-9

Plain DSP (essentia, no models):
  attack_logattacktime_ts   2 Hz  windowed (1 s / hop 0.5 s) LogAttackTime on the
  attack_tctototal_ts       2 Hz  window envelope; TCToTotal likewise
  attack_strongdecay_ts     2 Hz  StrongDecay on the raw window
  attack_maxratio_ts        2 Hz  AfterMaxToBeforeMaxEnergyRatio on the window's
                                  PitchYinFFT curve (it takes PITCH, not envelope);
                                  NaN where undefined (silent/unpitched window)
  stereo_width_ts         100 Hz  side/(mid+side) RMS ratio in [0,1] (0 = mono);
  stereo_corr_ts          100 Hz  per-frame L/R Pearson correlation. Mono source ->
                                  width 0 / corr 1 + meta note (needs the stereo
                                  read path; legacy pipeline downmixes).
  bark_bands_ts     (n, 27) 10 Hz BarkBands log-energies (dB)
  erb_bands_ts      (n, 40) 10 Hz ERBBands log-energies (dB)
  dissonance_ts             10 Hz Dissonance on spectral peaks
  pitch_salience_ts         10 Hz PitchSalience on the spectrum
  inharmonicity_ts          10 Hz Inharmonicity on harmonic peaks (0 where no f0)
  novelty_curve_ts          10 Hz NoveltyCurve over the Bark-band energies
  dyncomplexity_ts         0.2 Hz windowed (10 s / hop 5 s) DynamicComplexity
  dyncomplexity_loudness_ts 0.2 Hz  ... and its loudness estimate (dB)
  loudness_ebu_momentary_ts 10 Hz LoudnessEBUR128 momentary (LUFS; stereo path,
  loudness_ebu_shortterm_ts 10 Hz  mono gets the channel duplicated)
  chords_idx_ts            100 Hz ChordsDetection over the stored hpcp_ts; index
  chords_strength_ts       100 Hz  into CHORD_VOCAB (24 maj/min triads), -1 unknown
  chroma_linmap_ts    (n, 12) ~10 Hz NNLSChroma mid-range chromagram
  bass_chroma_linmap_ts (n,12) ~10 Hz  ... and its bass-range chromagram

Run under the mir venv (mir/bin/python — essentia+TF live there, NOT .venv).
Selftest (validates every extractor on one real track, prints shapes/rates):
  mir/bin/python src/spectral/whole_track_expanded.py <track_dir> [--seconds 90]
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# CPU-pin TF's CUDA probe before essentia (transitively TF) is imported.
# This is TF/CUDA only — HIP_VISIBLE_DEVICES stays untouched (MASTER §3: zeroing
# it crashes flash-attn/aiter imports elsewhere; and TF here is a CPU build anyway).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

logger = logging.getLogger(__name__)

MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"

MODEL_PATHS = {
    "maest": MODELS_ROOT / "essentia-zoo/feature-extractors/maest/discogs-maest-10s-fs-2.pb",
    "openl3": MODELS_ROOT / "essentia-zoo/feature-extractors/openl3/openl3-music-mel128-emb512-3.onnx",
    "effnet": MODELS_ROOT / "essentia-zoo/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre400": MODELS_ROOT / "essentia/genre_discogs400-discogs-effnet-1.pb",
    "moodtheme": MODELS_ROOT / "essentia/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "instrument": MODELS_ROOT / "essentia/mtg_jamendo_instrument-discogs-effnet-1.pb",
    "vggish": MODELS_ROOT / "essentia/audioset-vggish-3.pb",
    "deam": MODELS_ROOT / "essentia/deam-audioset-vggish-2.pb",
    "emomusic": MODELS_ROOT / "essentia/emomusic-audioset-vggish-2.pb",
}

# 24 maj/min triads; ChordsDetection emits e.g. "A", "Am", "C#", "C#m".
_ROOTS = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
CHORD_VOCAB = _ROOTS + [r + "m" for r in _ROOTS]
_CHORD_INDEX = {c: i for i, c in enumerate(CHORD_VOCAB)}
# ChordsDetection uses flats for some roots; fold enharmonics onto the vocab.
_ENHARMONIC = {"Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#"}

MODEL_FIELDS = [
    "maest_embed_ts",
    "effnet_genre400_ts", "effnet_moodtheme_ts", "effnet_instrument_ts",
    "va_deam_ts", "va_emomusic_ts",
]
DSP_FIELDS = [
    "attack_logattacktime_ts", "attack_tctototal_ts", "attack_strongdecay_ts",
    "attack_maxratio_ts",
    "stereo_width_ts", "stereo_corr_ts",
    "bark_bands_ts", "erb_bands_ts",
    "dissonance_ts", "pitch_salience_ts", "inharmonicity_ts",
    "novelty_curve_ts",
    "dyncomplexity_ts", "dyncomplexity_loudness_ts",
    "loudness_ebu_momentary_ts", "loudness_ebu_shortterm_ts",
    "chords_idx_ts", "chords_strength_ts",
    "chroma_linmap_ts", "bass_chroma_linmap_ts",
]
EXPANDED_FIELDS = MODEL_FIELDS + DSP_FIELDS
EXPANDED_VERSION = 1

_EPS = 1e-10


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    import essentia.standard as es
    return es.Resample(inputSampleRate=float(sr_in),
                       outputSampleRate=float(sr_out), quality=1)(
        audio.astype(np.float32))


class ExpandedExtractor:
    """One instance per worker process; TF models are built lazily on first use."""

    def __init__(self, models_root: Path = MODELS_ROOT,
                 enable_models: bool = True, enable_dsp: bool = True):
        self.models_root = Path(models_root)
        self.enable_models = enable_models
        self.enable_dsp = enable_dsp
        self._tf = {}          # name -> predictor (lazy)
        self._openl3_mel = None

    # ------------------------------------------------------------------ TF --

    def _predictor(self, name: str):
        if name in self._tf:
            return self._tf[name]
        import essentia.standard as es
        p = str(MODEL_PATHS[name])
        if name == "maest":
            # 10s model: patch = 626 mel frames (hop 256 @16 kHz ~= 10.0 s);
            # patchHopSize 313 -> ~5.008 s embedding hop. Layer 7 = the
            # metadata-recommended embedding output.
            alg = es.TensorflowPredictMAEST(
                graphFilename=p, output="PartitionedCall/Identity_7",
                patchHopSize=313)
        elif name == "effnet":
            alg = es.TensorflowPredictEffnetDiscogs(
                graphFilename=p, output="PartitionedCall:1")
        elif name == "vggish":
            alg = es.TensorflowPredictVGGish(
                graphFilename=p, output="model/vggish/embeddings")
        elif name == "genre400":
            alg = es.TensorflowPredict2D(
                graphFilename=p, input="serving_default_model_Placeholder",
                output="PartitionedCall:0")
        elif name in ("moodtheme", "instrument", "deam", "emomusic"):
            alg = es.TensorflowPredict2D(
                graphFilename=p, input="model/Placeholder",
                output="model/Sigmoid" if name in ("moodtheme", "instrument")
                else "model/Identity")
        elif name == "openl3":
            # The .pb graph is batchless and fights TensorflowPredict's 4D pool
            # tensors — use the ONNX twin on the explicitly CPU-pinned ORT EP.
            import onnxruntime as ort
            alg = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        else:
            raise KeyError(name)
        self._tf[name] = alg
        return alg

    def _maest(self, mono16: np.ndarray) -> np.ndarray:
        out = np.asarray(self._predictor("maest")(mono16))
        # (n_patches, 1, n_tokens, 768) or (n_patches, n_tokens, 768) -> token-mean
        out = out.reshape(out.shape[0], -1, out.shape[-1])
        return out.mean(axis=1).astype(np.float32)

    # NOTE: OpenL3 is NOT in the shipped field set (dropped by the gate-(a)
    # washout verdict, see module docstring). The working extractor below is
    # retained unwired for potential pool-item revival.
    def _openl3_patches(self, mono48: np.ndarray) -> np.ndarray:
        """Mel-spectrogram patches per the essentia OpenL3 reference extractor:
        48 kHz, frame 2048 / hop 242, 128 slaney mel bands, 1 s patches of
        exactly 199 frames, patch hop = OPENL3_HOP_S."""
        import essentia.standard as es
        if self._openl3_mel is None:
            self._openl3_mel = (
                es.Windowing(size=2048, normalized=False),
                es.Spectrum(size=2048),
                es.MelBands(highFrequencyBound=24000, inputSize=1025, log=False,
                            lowFrequencyBound=0, normalize="unit_tri",
                            numberBands=128, sampleRate=48000, type="magnitude",
                            warpingFormula="slaneyMel", weighting="linear"),
            )
        w, spec, mb = self._openl3_mel
        import essentia
        frames = [mb(spec(w(fr))) for fr in es.FrameGenerator(
            mono48, frameSize=2048, hopSize=242, startFromZero=True)]
        if len(frames) < 199:
            return np.zeros((0, 199, 128), dtype=np.float32)
        mel = np.asarray(frames, dtype=np.float32)          # (n_frames, 128)
        mel = 10.0 * np.log10(np.maximum(_EPS, mel))
        mel = np.maximum(mel, mel.max() - 80.0)
        hop_frames = int(round(OPENL3_HOP_S * 48000 / 242))
        starts = range(0, mel.shape[0] - 199 + 1, hop_frames)
        return np.stack([mel[s:s + 199] for s in starts]).astype(np.float32)

    def _openl3(self, mono48: np.ndarray) -> np.ndarray:
        patches = self._openl3_patches(mono48)
        if patches.shape[0] == 0:
            return np.zeros((0, 512), dtype=np.float32)
        sess = self._predictor("openl3")
        # ONNX input is (batch, 128 bands, 199 frames, 1)
        x = patches.transpose(0, 2, 1)[..., np.newaxis].astype(np.float32)
        outs = []
        for i in range(0, x.shape[0], 32):
            outs.append(sess.run(["embeddings"], {"melspectrogram": x[i:i + 32]})[0])
        return np.concatenate(outs, axis=0).astype(np.float32)

    # ----------------------------------------------------------------- DSP --

    @staticmethod
    def _windowed_attack(mono44: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        import essentia.standard as es
        win, hop = sr, sr // 2                        # 1 s window / 0.5 s hop -> 2 Hz
        env_alg = es.Envelope(sampleRate=sr)
        lat_alg = es.LogAttackTime(sampleRate=sr)
        tct_alg = es.TCToTotal()
        sd_alg = es.StrongDecay(sampleRate=sr)
        w_alg = es.Windowing(size=2048)
        spec_alg = es.Spectrum(size=2048)
        pitch_alg = es.PitchYinFFT(frameSize=2048, sampleRate=sr)
        n = max(0, (len(mono44) - win) // hop + 1)
        lat = np.full(n, np.nan, np.float32)
        tct = np.full(n, np.nan, np.float32)
        sd = np.full(n, np.nan, np.float32)
        amr = np.full(n, np.nan, np.float32)
        for i in range(n):
            seg = mono44[i * hop:i * hop + win]
            try:
                env = env_alg(seg)
                lat[i] = lat_alg(env)[0]
                tct[i] = tct_alg(env)
            except Exception:
                pass
            try:
                sd[i] = sd_alg(seg)
            except Exception:
                pass
            try:
                pitches = []
                for fr in es.FrameGenerator(seg, frameSize=2048, hopSize=1024,
                                            startFromZero=True):
                    p, conf = pitch_alg(spec_alg(w_alg(fr)))
                    pitches.append(p if conf > 0.3 else 0.0)
                pv = np.asarray(pitches, dtype=np.float32)
                if (pv > 0).any():
                    amr[i] = es.AfterMaxToBeforeMaxEnergyRatio()(pv)
            except Exception:
                pass
        return {"attack_logattacktime_ts": lat, "attack_tctototal_ts": tct,
                "attack_strongdecay_ts": sd, "attack_maxratio_ts": amr}

    @staticmethod
    def _stereo_fields(stereo44: Optional[np.ndarray], n_samples: int,
                       sr: int, frame_rate: int) -> Dict[str, np.ndarray]:
        hop = round(sr / frame_rate)
        n = max(1, n_samples // hop)
        if stereo44 is None:                          # mono source
            return {"stereo_width_ts": np.zeros(n, np.float32),
                    "stereo_corr_ts": np.ones(n, np.float32)}
        L, R = stereo44[:, 0], stereo44[:, 1]
        width = np.zeros(n, np.float32)
        corr = np.ones(n, np.float32)
        for i in range(n):
            l, r = L[i * hop:(i + 1) * hop], R[i * hop:(i + 1) * hop]
            mid, side = (l + r) * 0.5, (l - r) * 0.5
            m_rms = float(np.sqrt(np.mean(mid ** 2)))
            s_rms = float(np.sqrt(np.mean(side ** 2)))
            # bounded width in [0,1]: 0 = mono, 0.5 = decorrelated, 1 = all-side
            width[i] = s_rms / (m_rms + s_rms + _EPS)
            sl, sr_ = float(l.std()), float(r.std())
            if sl > _EPS and sr_ > _EPS:
                corr[i] = float(np.corrcoef(l, r)[0, 1])
        return {"stereo_width_ts": width, "stereo_corr_ts": corr}

    def _spectral_10hz(self, mono44: np.ndarray, sr: int) -> Tuple[Dict[str, np.ndarray], float]:
        """Bark/ERB bands, dissonance, pitch salience, inharmonicity, novelty —
        one 10 Hz frame loop shares the spectrum."""
        import essentia.standard as es
        hop = sr // 10                                # 10 Hz
        frame_size = 2048
        w = es.Windowing(size=frame_size)
        spec = es.Spectrum(size=frame_size)
        bark = es.BarkBands(sampleRate=sr, numberBands=27)
        erb = es.ERBBands(sampleRate=sr, inputSize=frame_size // 2 + 1)
        peaks = es.SpectralPeaks(maxPeaks=100, magnitudeThreshold=1e-5,
                                 sampleRate=sr, orderBy="frequency")
        diss = es.Dissonance()
        psal = es.PitchSalience(sampleRate=sr)
        pitch = es.PitchYinFFT(frameSize=frame_size, sampleRate=sr)
        harm = es.HarmonicPeaks()
        inh = es.Inharmonicity()
        rows_bark, rows_erb, rows_d, rows_ps, rows_in = [], [], [], [], []
        for fr in es.FrameGenerator(mono44, frameSize=frame_size, hopSize=hop,
                                    startFromZero=True):
            s = spec(w(fr))
            b = bark(s)
            rows_bark.append(b)
            rows_erb.append(erb(s))
            f, m = peaks(s)
            try:
                rows_d.append(diss(f, m))
            except Exception:
                rows_d.append(0.0)
            rows_ps.append(psal(s))
            try:
                p0, conf = pitch(s)
                if conf > 0.3 and p0 > 0 and len(f) and f[0] > 0:
                    hf, hm = harm(f, m, p0)
                    rows_in.append(inh(hf, hm))
                else:
                    rows_in.append(0.0)
            except Exception:
                rows_in.append(0.0)
        bark_arr = np.asarray(rows_bark, dtype=np.float32)
        out = {
            "bark_bands_ts": (10 * np.log10(np.maximum(_EPS, bark_arr))).astype(np.float32),
            "erb_bands_ts": (10 * np.log10(np.maximum(_EPS, np.asarray(rows_erb, np.float32)))).astype(np.float32),
            "dissonance_ts": np.asarray(rows_d, np.float32),
            "pitch_salience_ts": np.asarray(rows_ps, np.float32),
            "inharmonicity_ts": np.asarray(rows_in, np.float32),
        }
        try:
            nov = np.asarray(es.NoveltyCurve(frameRate=10.0)(bark_arr), np.float32)
            # raw scale is meaningless (grows with track energy) — per-track
            # max-normalise; novelty is inherently a relative measure
            out["novelty_curve_ts"] = nov / max(float(nov.max()), _EPS)
        except Exception as e:
            logger.warning(f"  novelty curve failed: {e}")
        return out, 10.0

    @staticmethod
    def _dyncomplexity(mono44: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        import essentia.standard as es
        win, hop = sr * 10, sr * 5                    # 10 s window / 5 s hop -> 0.2 Hz
        alg = es.DynamicComplexity(sampleRate=sr)
        n = max(0, (len(mono44) - win) // hop + 1)
        dc = np.zeros(n, np.float32)
        ld = np.zeros(n, np.float32)
        for i in range(n):
            try:
                dc[i], ld[i] = alg(mono44[i * hop:i * hop + win])
            except Exception:
                pass
        return {"dyncomplexity_ts": dc, "dyncomplexity_loudness_ts": ld}

    @staticmethod
    def _ebu(stereo44: Optional[np.ndarray], mono44: np.ndarray,
             sr: int) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        import essentia.standard as es
        st = stereo44 if stereo44 is not None else np.stack([mono44, mono44], axis=1)
        mom, short, integrated, lrange = es.LoudnessEBUR128(
            hopSize=0.1, sampleRate=sr)(st.astype(np.float32))
        return ({"loudness_ebu_momentary_ts": np.asarray(mom, np.float32),
                 "loudness_ebu_shortterm_ts": np.asarray(short, np.float32)},
                {"loudness_ebu_integrated": float(integrated),
                 "loudness_ebu_range": float(lrange)})

    @staticmethod
    def _chords(hpcp: np.ndarray, hpcp_rate: float, sr: int) -> Dict[str, np.ndarray]:
        import essentia.standard as es
        hop = int(round(sr / hpcp_rate))
        chords, strength = es.ChordsDetection(
            hopSize=hop, sampleRate=sr, windowSize=2.0)(hpcp.astype(np.float32))
        idx = np.full(len(chords), -1.0, np.float32)
        for i, c in enumerate(chords):
            root = c[:-1] if c.endswith("m") else c
            root = _ENHARMONIC.get(root, root)
            name = root + "m" if c.endswith("m") else root
            idx[i] = _CHORD_INDEX.get(name, -1)
        return {"chords_idx_ts": idx,
                "chords_strength_ts": np.asarray(strength, np.float32)}

    def _nnls(self, mono44: np.ndarray, sr: int) -> Tuple[Dict[str, np.ndarray], float]:
        import essentia.standard as es
        # frame 16384 for low-frequency resolution (the bass chromagram needs it).
        # useNNLS=False: this essentia build's NNLS solver returns all-zero
        # semitone/chroma outputs (verified empirically 2026-07-14); the linear
        # spectral mapping shares the same tuned log-freq frontend and works.
        frame_size = 16384
        hop = sr // 10                                # ~10 Hz
        w = es.Windowing(size=frame_size)
        spec = es.Spectrum(size=frame_size)
        logspec = es.LogSpectrum(frameSize=frame_size // 2 + 1)
        frames, mean_tunings, local_tunings = [], [], []
        for fr in es.FrameGenerator(mono44, frameSize=frame_size, hopSize=hop,
                                    startFromZero=True):
            lf, mt, lt = logspec(spec(w(fr)))
            frames.append(lf)
            mean_tunings.append(mt)
            local_tunings.append(lt)
        if not frames:
            return {}, 10.0
        import essentia
        _tuned, _semi, bass, chroma = es.NNLSChroma(
            frameSize=frame_size // 2 + 1, useNNLS=False)(
            essentia.array(frames),
            essentia.array(np.mean(mean_tunings, axis=0)),
            essentia.array(local_tunings))
        return ({"chroma_linmap_ts": np.asarray(chroma, np.float32),
                 "bass_chroma_linmap_ts": np.asarray(bass, np.float32)},
                sr / hop)

    # ------------------------------------------------------------- driver --

    def extract(self, full_mix: Path, wanted: Optional[List[str]] = None,
                existing: Optional[Dict[str, np.ndarray]] = None,
                existing_meta: Optional[Dict] = None,
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict]:
        """Compute expanded fields for one track.

        wanted: field subset (None = all enabled); existing/existing_meta let
        chord detection reuse the stored hpcp_ts. Returns (data, field_rates,
        extra_meta).
        """
        from core.file_utils import read_audio
        wanted = set(wanted if wanted is not None else EXPANDED_FIELDS)
        if not self.enable_models:
            wanted -= set(MODEL_FIELDS)
        if not self.enable_dsp:
            wanted -= set(DSP_FIELDS)

        raw, sr = read_audio(str(full_mix))
        stereo = raw.astype(np.float32) if raw.ndim == 2 and raw.shape[1] == 2 else None
        mono = raw.mean(axis=1).astype(np.float32) if raw.ndim > 1 else raw.astype(np.float32)
        mono44 = _resample(mono, sr, 44100)
        stereo44 = None
        if stereo is not None:
            stereo44 = np.stack([_resample(stereo[:, 0], sr, 44100),
                                 _resample(stereo[:, 1], sr, 44100)], axis=1)

        data: Dict[str, np.ndarray] = {}
        rates: Dict[str, float] = {}
        extra: Dict = {"expanded_version": EXPANDED_VERSION}

        def put(d: Dict[str, np.ndarray], fps):
            for k, v in d.items():
                if k in wanted:
                    data[k] = v
                    rates[k] = float(fps if not isinstance(fps, dict) else fps[k])

        # --- DSP -----------------------------------------------------------
        if wanted & {"attack_logattacktime_ts", "attack_tctototal_ts",
                     "attack_strongdecay_ts", "attack_maxratio_ts"}:
            put(self._windowed_attack(mono44, 44100), 2.0)
        if wanted & {"stereo_width_ts", "stereo_corr_ts"}:
            put(self._stereo_fields(stereo44, len(mono44), 44100, 100), 100.0)
            extra["stereo_source"] = stereo is not None
        if wanted & {"bark_bands_ts", "erb_bands_ts", "dissonance_ts",
                     "pitch_salience_ts", "inharmonicity_ts", "novelty_curve_ts"}:
            d10, fps10 = self._spectral_10hz(mono44, 44100)
            put(d10, fps10)
        if wanted & {"dyncomplexity_ts", "dyncomplexity_loudness_ts"}:
            put(self._dyncomplexity(mono44, 44100), 0.2)
        if wanted & {"loudness_ebu_momentary_ts", "loudness_ebu_shortterm_ts"}:
            debu, scal = self._ebu(stereo44, mono44, 44100)
            put(debu, 10.0)
            extra.update(scal)
        if wanted & {"chords_idx_ts", "chords_strength_ts"}:
            hpcp = existing.get("hpcp_ts") if existing else None
            if hpcp is not None and len(hpcp):
                hpcp_rate = float((existing_meta or {}).get("frame_rate", 100))
                put(self._chords(hpcp, hpcp_rate, 44100), hpcp_rate)
                extra["chord_vocab"] = CHORD_VOCAB
            else:
                logger.warning("  chords skipped: no hpcp_ts available")
        if wanted & {"chroma_linmap_ts", "bass_chroma_linmap_ts"}:
            dnn, fpsnn = self._nnls(mono44, 44100)
            put(dnn, fpsnn)

        # --- models ----------------------------------------------------------
        needs16 = wanted & {"effnet_genre400_ts", "effnet_moodtheme_ts",
                            "effnet_instrument_ts", "va_deam_ts", "va_emomusic_ts",
                            "maest_embed_ts"}
        mono16 = _resample(mono44, 44100, 16000) if needs16 else None
        if "maest_embed_ts" in wanted:
            emb = self._maest(mono16)
            put({"maest_embed_ts": emb}, 16000.0 / (313 * 256))
            extra["maest_model"] = MODEL_PATHS["maest"].name
        if wanted & {"effnet_genre400_ts", "effnet_moodtheme_ts", "effnet_instrument_ts"}:
            emb = np.asarray(self._predictor("effnet")(mono16), dtype=np.float32)
            fps = 16000.0 / (62 * 256)
            if "effnet_genre400_ts" in wanted:
                put({"effnet_genre400_ts": np.asarray(
                    self._predictor("genre400")(emb), np.float32)}, fps)
            if "effnet_moodtheme_ts" in wanted:
                put({"effnet_moodtheme_ts": np.asarray(
                    self._predictor("moodtheme")(emb), np.float32)}, fps)
            if "effnet_instrument_ts" in wanted:
                put({"effnet_instrument_ts": np.asarray(
                    self._predictor("instrument")(emb), np.float32)}, fps)
        if wanted & {"va_deam_ts", "va_emomusic_ts"}:
            vemb = np.asarray(self._predictor("vggish")(mono16), dtype=np.float32)
            fps = 16000.0 / (96 * 160)
            if "va_deam_ts" in wanted:
                put({"va_deam_ts": np.asarray(
                    self._predictor("deam")(vemb), np.float32)}, fps)
            if "va_emomusic_ts" in wanted:
                put({"va_emomusic_ts": np.asarray(
                    self._predictor("emomusic")(vemb), np.float32)}, fps)

        return data, rates, extra


OPENL3_HOP_S = 2.0     # OpenL3 patch hop (s) — coarse per C's size-budget call


def merge_expanded(npz_path: Path, data: Dict[str, np.ndarray],
                   rates: Dict[str, float], extra: Dict) -> None:
    """Merge expanded fields into an existing sidecar (atomic write)."""
    from spectral.whole_track_timeseries import load_timeseries_npz
    old, meta = load_timeseries_npz(npz_path)
    old.update({k: v.astype(np.float32) for k, v in data.items()})
    meta["fields"] = sorted(old.keys())
    meta.setdefault("field_rates", {}).update(rates)
    meta.setdefault("expanded", {}).update(extra)
    payload = dict(old)
    payload["__meta__"] = np.array(json.dumps(meta))
    # tmp name must END in .npz — np.savez appends the extension otherwise,
    # leaving 'x.tmp.npz' while os.replace looks for 'x.tmp' (pilot bug 2026-07-14)
    tmp = npz_path.parent / (npz_path.stem + ".tmp.npz")
    np.savez_compressed(str(tmp), **payload)
    os.replace(tmp, npz_path)


def missing_expanded_fields(npz_path: Path,
                            wanted: Optional[List[str]] = None) -> List[str]:
    """Which of the wanted expanded fields are absent from an existing sidecar."""
    wanted = list(wanted if wanted is not None else EXPANDED_FIELDS)
    try:
        with np.load(str(npz_path), allow_pickle=False) as z:
            have = set(z.files)
    except Exception:
        return wanted
    return [f for f in wanted if f not in have]


# --------------------------------------------------------------- selftest --

def _selftest():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    ap = argparse.ArgumentParser(description="Run every expanded extractor on one track dir")
    ap.add_argument("track_dir", type=Path)
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="truncate audio to this length (0 = full track)")
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from spectral.whole_track_timeseries import find_full_mix, load_timeseries_npz
    full_mix = find_full_mix(args.track_dir) if args.track_dir.is_dir() else args.track_dir
    if full_mix is None:
        sys.exit(f"no full_mix in {args.track_dir}")
    print(f"selftest on {full_mix} (first {args.seconds or 'all'} s)")

    src = full_mix
    if args.seconds:
        # truncate via a temp wav so every extractor sees the same short input
        from core.file_utils import read_audio
        import soundfile as sf
        raw, sr = read_audio(str(full_mix))
        raw = raw[:int(args.seconds * sr)]
        src = Path("/tmp/claude-1000/_expanded_selftest.wav")
        src.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(src), raw, sr)

    existing, existing_meta = None, None
    npz = (args.track_dir / f"{args.track_dir.name}.TIMESERIES.npz") \
        if args.track_dir.is_dir() else None
    if npz and npz.exists():
        existing, existing_meta = load_timeseries_npz(npz)
        # slice the stored hpcp to the truncated duration so chords line up
        if args.seconds and existing is not None and "hpcp_ts" in existing:
            n = int(args.seconds * existing_meta.get("frame_rate", 100))
            existing = {"hpcp_ts": existing["hpcp_ts"][:n]}
        print(f"reusing hpcp from {npz.name}" if existing else "no sidecar hpcp")

    ex = ExpandedExtractor(enable_models=not args.no_models)
    t0 = time.time()
    data, rates, extra = ex.extract(src, existing=existing, existing_meta=existing_meta)
    dt = time.time() - t0
    print(f"\n{len(data)} fields in {dt:.1f}s:")
    for k in sorted(data):
        v = data[k]
        print(f"  {k:28s} {str(v.shape):14s} @ {rates[k]:8.3f} Hz  "
              f"[{np.nanmin(v):9.3f} .. {np.nanmax(v):9.3f}]  "
              f"nan={int(np.isnan(v).sum())}")
    missing = [f for f in EXPANDED_FIELDS if f not in data]
    if missing:
        print(f"\nMISSING: {missing}")
    print(f"\nextra meta: { {k: (v if not isinstance(v, list) else f'[{len(v)} items]') for k, v in extra.items()} }")


if __name__ == "__main__":
    _selftest()
