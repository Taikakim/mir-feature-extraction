import sys
import os
import glob
import json
import threading
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSlider, QPushButton, QLabel, QRadioButton, QButtonGroup,
    QGroupBox, QFormLayout, QCheckBox, QStackedWidget, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
import numpy as np
import sounddevice as sd
import soundfile as sf

DATASET_DIR = "/run/media/kim/Mantu/ai-music/Goa_Separated_crops"
SCORES_FILE = "pitch_shift_scores.csv"
STRETCH_SCORES_FILE = "stretch_scores.csv"

class AudioEngine(QObject):
    # Signals for updating UI
    playback_position_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.stream = None
        self.current_buffer = None
        self.is_playing = False
        self.playhead = 0
        self._pending = None  # (data,) set by render threads, consumed by callback

        # We will hold pre-rendered buffers here
        self.buffers = {
            "Original": None,
            "Bungee": None,
            "Rubberband": None,
            "Pedalboard": None,
            "Sox": None,
        }
        self.active_algorithm = "Original"

    def set_buffer(self, algo_name, data, samplerate):
        self.buffers[algo_name] = data
        self.samplerate = samplerate
        if algo_name == self.active_algorithm:
            if self.stream is not None and self.stream.active:
                # Callback is running — queue the swap so it happens inside the callback
                self._pending = data
            else:
                # Stream not running — safe to set directly
                self.current_buffer = data

    def switch_algorithm(self, algo_name):
        self.active_algorithm = algo_name
        buf = self.buffers.get(algo_name)
        if buf is not None:
            if self.stream is not None and self.stream.active:
                self._pending = buf
            else:
                self.current_buffer = buf

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)

        # Apply any pending buffer swap — safe here since we're in the callback thread
        pending = self._pending
        if pending is not None:
            self._pending = None
            old = self.current_buffer
            if old is not None and len(old) > 0:
                rel_pos = self.playhead / len(old)
                self.current_buffer = pending
                new_len = len(pending)
                self.playhead = max(0, min(int(rel_pos * new_len), new_len - 1)) if new_len > 0 else 0
            else:
                self.current_buffer = pending

        current_buf = self.current_buffer
        if not self.is_playing or current_buf is None or len(current_buf) == 0:
            outdata[:] = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
            return
            
        buffer_len = len(current_buf)
        if self.playhead >= buffer_len:
            self.playhead = 0
        
        # Handle looping
        remaining = frames
        out_idx = 0
        while remaining > 0:
            chunk = min(remaining, buffer_len - self.playhead)
            outdata[out_idx:out_idx+chunk] = current_buf[self.playhead:self.playhead+chunk]
            self.playhead += chunk
            out_idx += chunk
            remaining -= chunk
            if self.playhead >= buffer_len:
                self.playhead = 0

    def start(self):
        if self.stream is None and self.current_buffer is not None:
            channels = self.current_buffer.shape[1] if len(self.current_buffer.shape) > 1 else 1
            self.stream = sd.OutputStream(
                samplerate=self.samplerate, 
                channels=channels,
                callback=self._audio_callback
            )
            self.stream.start()
        self.is_playing = True

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False
        self.playhead = 0
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

class PitchShiftGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Shift Algorithm Comparison")
        self.resize(800, 600)
        
        self.audio_engine = AudioEngine()
        self.tracks = self._scan_dataset()
        
        self.init_ui()

    def _scan_dataset(self):
        """Scans DATASET_DIR for tracks and returns a dict mapping track name to path."""
        tracks = {}
        if not os.path.exists(DATASET_DIR):
            print(f"Dataset directory not found: {DATASET_DIR}")
            return tracks
            
        for d in os.listdir(DATASET_DIR):
            full_path = os.path.join(DATASET_DIR, d)
            if os.path.isdir(full_path):
                # Check if it has flac files
                if glob.glob(os.path.join(full_path, "*.flac")):
                    tracks[d] = full_path
        return tracks

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 1. Track Selection
        track_layout = QHBoxLayout()
        track_layout.addWidget(QLabel("Track:"))
        self.track_combo = QComboBox()
        self.track_combo.addItems(sorted(list(self.tracks.keys())))
        self.track_combo.currentTextChanged.connect(self.on_track_changed)
        track_layout.addWidget(self.track_combo, 1)
        
        self.crop_combo = QComboBox()
        self.crop_combo.currentTextChanged.connect(self.on_crop_changed)
        track_layout.addWidget(QLabel("Crop:"))
        track_layout.addWidget(self.crop_combo)
        
        self.pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_slider.setMinimum(0)
        self.pos_slider.setMaximum(1000)
        self.pos_slider.setValue(0)
        self.pos_label = QLabel("Pos: 0.000")
        self.pos_slider.sliderReleased.connect(self.on_pos_released)
        self.pos_slider.valueChanged.connect(self.on_pos_value_changed)
        
        track_layout.addWidget(QLabel("Position:"))
        track_layout.addWidget(self.pos_slider)
        track_layout.addWidget(self.pos_label)
        
        main_layout.addLayout(track_layout)

        # 2. Controls (Pitch, Stretch)
        controls_group = QGroupBox("Shift Parameters")
        controls_layout = QFormLayout()

        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setMinimum(-12)
        self.pitch_slider.setMaximum(12)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pitch_slider.setTickInterval(1)
        self.pitch_label = QLabel("0 semitones")
        self.pitch_slider.valueChanged.connect(self.on_pitch_value_changed)
        self.pitch_slider.sliderReleased.connect(self.on_parameter_released)
        pitch_box = QHBoxLayout()
        pitch_box.addWidget(self.pitch_slider)
        pitch_box.addWidget(self.pitch_label)
        controls_layout.addRow("Pitch Shift:", pitch_box)

        self.stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.stretch_slider.setMinimum(50)
        self.stretch_slider.setMaximum(200)
        self.stretch_slider.setValue(100)
        self.stretch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.stretch_slider.setTickInterval(10)
        self.stretch_label = QLabel("x1.00")
        self.stretch_slider.valueChanged.connect(self.on_stretch_value_changed)
        self.stretch_slider.sliderReleased.connect(self.on_parameter_released)
        stretch_box = QHBoxLayout()
        stretch_box.addWidget(self.stretch_slider)
        stretch_box.addWidget(self.stretch_label)
        controls_layout.addRow("Time Stretch:", stretch_box)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # 3. Algorithm Selection
        algo_group = QGroupBox("Algorithm (1-5)")
        algo_layout = QHBoxLayout()
        self.algo_btn_group = QButtonGroup(self)

        algos = ["Original", "Bungee", "Rubberband", "Pedalboard", "Sox"]
        self.algo_radios = {}
        for i, algo in enumerate(algos):
            rb = QRadioButton(f"{algo} ({i+1})")
            if i == 0:
                rb.setChecked(True)
            algo_layout.addWidget(rb)
            self.algo_btn_group.addButton(rb, i)
            self.algo_radios[algo] = rb

        self.algo_btn_group.idClicked.connect(self.on_algo_changed)
        algo_group.setLayout(algo_layout)
        main_layout.addWidget(algo_group)

        # 3b. Per-algorithm options (stacked, switches with algo selection)
        options_group = QGroupBox("Algorithm Options")
        options_layout = QVBoxLayout()
        self.algo_options_stack = QStackedWidget()

        def _page(widgets):
            """Helper: wrap a list of widgets into a horizontal QWidget page."""
            p = QWidget()
            l = QHBoxLayout(p)
            l.setContentsMargins(0, 0, 0, 0)
            for w in widgets:
                l.addWidget(w)
            l.addStretch()
            return p

        # Page 0 — Original
        self.algo_options_stack.addWidget(_page([QLabel("Reference — no options")]))

        # Page 1 — Bungee
        self.algo_options_stack.addWidget(_page([QLabel("No quality/formant controls exposed by bungee-python")]))

        # Page 2 — Rubberband
        self.rb_formants_cb = QCheckBox("Preserve Formants")
        self.rb_engine_combo = QComboBox()
        self.rb_engine_combo.addItems(["R2 (faster)", "R3 (finer, rubberband ≥3)"])
        self.rb_crisp_spin = QSpinBox()
        self.rb_crisp_spin.setRange(0, 6)
        self.rb_crisp_spin.setValue(5)
        self.rb_crisp_spin.setToolTip("Crisp level 0–6 (R2 only; default=5)")
        self.algo_options_stack.addWidget(_page([
            self.rb_formants_cb,
            QLabel("  Engine:"), self.rb_engine_combo,
            QLabel("  Crisp (R2, 0-6):"), self.rb_crisp_spin,
        ]))

        # Page 3 — Pedalboard
        self.pb_formants_cb = QCheckBox("Preserve Formants")
        self.pb_transient_combo = QComboBox()
        self.pb_transient_combo.addItems(["crisp", "mixed", "smooth"])
        self.pb_detector_combo = QComboBox()
        self.pb_detector_combo.addItems(["compound", "percussive", "soft"])
        self.algo_options_stack.addWidget(_page([
            self.pb_formants_cb,
            QLabel("  Transients:"), self.pb_transient_combo,
            QLabel("  Detector:"), self.pb_detector_combo,
        ]))

        # Page 4 — Sox
        self.sox_wsola_cb = QCheckBox("WSOLA stretch")
        self.sox_wsola_cb.setChecked(True)
        self.sox_audio_type_combo = QComboBox()
        self.sox_audio_type_combo.addItems(["Music (m)", "Speech (s)", "Linear (l)"])
        self.sox_segment_spin = QSpinBox()
        self.sox_segment_spin.setRange(10, 200)
        self.sox_segment_spin.setValue(82)
        self.sox_segment_spin.setSuffix(" ms")
        self.sox_search_spin = QSpinBox()
        self.sox_search_spin.setRange(0, 100)
        self.sox_search_spin.setValue(14)
        self.sox_search_spin.setSuffix(" ms")
        self.sox_pitch_quality_combo = QComboBox()
        self.sox_pitch_quality_combo.addItems(["High (default)", "Quick cubic (-q)"])
        self.algo_options_stack.addWidget(_page([
            self.sox_wsola_cb,
            QLabel("  Type:"), self.sox_audio_type_combo,
            QLabel("  Seg:"), self.sox_segment_spin,
            QLabel("  Search:"), self.sox_search_spin,
            QLabel("  Pitch quality:"), self.sox_pitch_quality_combo,
        ]))



        self.algo_btn_group.idClicked.connect(self.algo_options_stack.setCurrentIndex)

        # Re-render only the affected algorithm when its options change
        self.rb_formants_cb.stateChanged.connect(lambda: self._rerender_one(self._render_rubberband))
        self.rb_engine_combo.currentIndexChanged.connect(lambda: self._rerender_one(self._render_rubberband))
        self.rb_crisp_spin.valueChanged.connect(lambda: self._rerender_one(self._render_rubberband))
        self.pb_formants_cb.stateChanged.connect(lambda: self._rerender_one(self._render_pedalboard))
        self.pb_transient_combo.currentIndexChanged.connect(lambda: self._rerender_one(self._render_pedalboard))
        self.pb_detector_combo.currentIndexChanged.connect(lambda: self._rerender_one(self._render_pedalboard))
        self.sox_wsola_cb.stateChanged.connect(lambda: self._rerender_one(self._render_sox))
        self.sox_audio_type_combo.currentIndexChanged.connect(lambda: self._rerender_one(self._render_sox))
        self.sox_pitch_quality_combo.currentIndexChanged.connect(lambda: self._rerender_one(self._render_sox))
        # Debounce spinboxes: only re-render 300ms after the user stops spinning
        self._sox_debounce = QTimer()
        self._sox_debounce.setSingleShot(True)
        self._sox_debounce.setInterval(300)
        self._sox_debounce.timeout.connect(lambda: self._rerender_one(self._render_sox))
        self.sox_segment_spin.valueChanged.connect(self._sox_debounce.start)
        self.sox_search_spin.valueChanged.connect(self._sox_debounce.start)

        options_layout.addWidget(self.algo_options_stack)
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # 4. Transport & Scoring
        transport_layout = QVBoxLayout()

        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        play_layout.addWidget(self.play_btn)
        play_layout.addStretch()
        transport_layout.addLayout(play_layout)

        # Score buttons: algo-named, one per non-Original algorithm; clicking votes for that algo
        score_algos = ["Bungee", "Rubberband", "Pedalboard", "Sox"]

        score_layout_pitch = QHBoxLayout()
        score_layout_pitch.addWidget(QLabel("Best pitch shift:"))
        for algo in score_algos:
            btn = QPushButton(algo)
            btn.setProperty("score_algo", algo)
            btn.clicked.connect(self.on_pitch_score_clicked)
            score_layout_pitch.addWidget(btn)
        transport_layout.addLayout(score_layout_pitch)

        score_layout_stretch = QHBoxLayout()
        score_layout_stretch.addWidget(QLabel("Best time stretch:"))
        for algo in score_algos:
            btn = QPushButton(algo)
            btn.setProperty("score_algo", algo)
            btn.clicked.connect(self.on_stretch_score_clicked)
            score_layout_stretch.addWidget(btn)
        transport_layout.addLayout(score_layout_stretch)
        
        main_layout.addLayout(transport_layout)
        
        # Initial populate
        if self.tracks:
            self.on_track_changed(self.track_combo.currentText())

    def on_track_changed(self, track_name):
        if not track_name: return
        track_path = self.tracks[track_name]
        
        info_files = glob.glob(os.path.join(track_path, "*.INFO"))
        self.crop_positions = []
        for f in info_files:
            basename = os.path.basename(f)
            parts = basename.rsplit('_', 1)
            if len(parts) == 2:
                idx_part = parts[1].replace('.INFO', '')
                if idx_part.isdigit():
                    idx = int(idx_part)
                    try:
                        with open(f, 'r') as infof:
                            data = json.load(infof)
                        pos = data.get("position", 0.0)
                        self.crop_positions.append((pos, idx))
                    except Exception as e:
                        print(f"Error reading info {f}: {e}")
        
        self.crop_positions.sort(key=lambda x: x[0])
        
        self.crop_combo.blockSignals(True)
        self.crop_combo.clear()
        self.crop_combo.addItems([str(c[1]) for c in self.crop_positions])
        self.crop_combo.blockSignals(False)
        
        if self.crop_positions:
            self.on_crop_changed(str(self.crop_positions[0][1]))

    def on_crop_changed(self, crop_idx_str):
        if not hasattr(self, 'crop_positions'): return
        if not crop_idx_str: return
        crop_idx = int(crop_idx_str)
        # Find pos
        pos = 0.0
        for p, idx in self.crop_positions:
            if idx == crop_idx:
                pos = p
                break
                
        self.pos_slider.blockSignals(True)
        self.pos_slider.setValue(int(pos * 1000))
        self.pos_label.setText(f"Pos: {pos:.3f}")
        self.pos_slider.blockSignals(False)
        
        self.load_crop(crop_idx)

    def on_pos_value_changed(self, value):
        self.pos_label.setText(f"Pos: {value/1000.:.3f}")

    def on_pos_released(self):
        if not hasattr(self, 'crop_positions') or not self.crop_positions: return
        val = self.pos_slider.value() / 1000.0
        # find closest crop index
        closest_idx = min(self.crop_positions, key=lambda x: abs(x[0] - val))[1]
        
        # update crop combo, which will trigger load_crop via signals
        idx_str = str(closest_idx)
        if self.crop_combo.currentText() != idx_str:
            self.crop_combo.setCurrentText(idx_str)

    def on_pitch_value_changed(self, value):
        self.pitch_label.setText(f"{value} semitones")

    def on_stretch_value_changed(self, value):
        self.stretch_label.setText(f"x{value / 100.0:.2f}")

    def on_parameter_released(self):
        # Trigger re-render now that the user has released a slider
        if self.audio_engine.current_buffer is not None:
            original_data = self.audio_engine.buffers["Original"]
            if original_data is not None:
                pitch_val = self.pitch_slider.value()
                stretch_val = self.stretch_slider.value() / 100.0
                self.render_transformations(original_data, self.audio_engine.samplerate, pitch_val, stretch_val)

    def _rerender_one(self, render_fn):
        """Re-render a single algorithm using current pitch/stretch values."""
        original_data = self.audio_engine.buffers.get("Original")
        if original_data is None:
            return
        pitch_val = self.pitch_slider.value()
        stretch_val = self.stretch_slider.value() / 100.0
        if pitch_val == 0 and stretch_val == 1.0:
            return  # passthrough — buffer already set to original on load
        sr = self.audio_engine.samplerate
        threading.Thread(target=render_fn, args=(original_data, sr, pitch_val, stretch_val)).start()

    def on_algo_changed(self, btn_id):
        algos = ["Original", "Bungee", "Rubberband", "Pedalboard", "Sox"]
        algo_name = algos[btn_id]
        self.audio_engine.switch_algorithm(algo_name)

    def toggle_playback(self):
        if self.audio_engine.is_playing:
            self.audio_engine.pause()
            self.play_btn.setText("Play")
        else:
            self.audio_engine.start()
            self.play_btn.setText("Pause")

    def on_pitch_score_clicked(self):
        chosen = self.sender().property("score_algo")
        track = self.track_combo.currentText()
        crop = self.crop_combo.currentText()
        pitch = self.pitch_slider.value()
        stretch = self.stretch_slider.value() / 100.0

        line = f"{track},{crop},{pitch},{stretch},{chosen}\n"
        with open(SCORES_FILE, 'a') as f:
            f.write(line)
        print(f"Saved pitch vote: {line.strip()}")

    def on_stretch_score_clicked(self):
        chosen = self.sender().property("score_algo")
        track = self.track_combo.currentText()
        crop = self.crop_combo.currentText()
        pitch = self.pitch_slider.value()
        stretch = self.stretch_slider.value() / 100.0

        line = f"{track},{crop},{stretch},{pitch},{chosen}\n"
        with open(STRETCH_SCORES_FILE, 'a') as f:
            f.write(line)
        print(f"Saved stretch vote: {line.strip()}")

    def find_zero_crossing(self, data, sample_idx):
        if len(data.shape) > 1:
            data_mono = data.mean(axis=1) # mix to mono for zero crossing detection
        else:
            data_mono = data
            
        idx = int(sample_idx)
        if idx >= len(data_mono) - 1:
            return len(data_mono) - 1
        if idx <= 0:
            return 0
            
        # search within a small window, say 100ms
        search_radius = 4410
        start_search = max(1, idx - search_radius)
        end_search = min(len(data_mono) - 1, idx + search_radius)
        
        best_idx = idx
        min_dist = float('inf')
        
        # Only pick positive-going zero crossings to guarantee phase continuity
        for i in range(start_search, end_search):
            if data_mono[i-1] <= 0 and data_mono[i] > 0:
                dist = abs(i - idx)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        return best_idx

    def load_crop(self, crop_idx):
        if not self.track_combo.currentText(): return
        
        track_path = self.tracks[self.track_combo.currentText()]
        track_name = os.path.basename(track_path)
        
        crop_base = f"{track_name}_{crop_idx}"
        audio_file = os.path.join(track_path, f"{crop_base}.flac")
        downbeats_file = os.path.join(track_path, f"{crop_base}.DOWNBEATS")
        
        if not os.path.exists(audio_file) or not os.path.exists(downbeats_file):
            print(f"Missing audio or downbeats for {crop_base}")
            return
            
        data, sr = sf.read(audio_file)
        
        with open(downbeats_file, 'r') as f:
            downbeats = [float(line.strip()) for line in f.readlines() if line.strip()]
            
        if len(downbeats) < 2:
            print("Not enough downbeats for a loop")
            return
            
        start_time = downbeats[0]
        end_time = downbeats[-1]
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        start_sample = self.find_zero_crossing(data, start_sample)
        end_sample = self.find_zero_crossing(data, end_sample)
        
        loop_data = data[start_sample:end_sample]
        
        self.audio_engine.pause()
        self.audio_engine.set_buffer("Original", loop_data, sr)
        
        # Reset other buffers
        self.audio_engine.buffers["Bungee"] = None
        self.audio_engine.buffers["Rubberband"] = None
        self.audio_engine.buffers["Pedalboard"] = None
        self.audio_engine.buffers["Sox"] = None
        
        # Trigger rendering
        pitch_val = self.pitch_slider.value()
        stretch_val = self.stretch_slider.value() / 100.0
        self.render_transformations(loop_data, sr, pitch_val, stretch_val)
        
        if self.play_btn.text() == "Pause":
            self.audio_engine.start()

    def render_transformations(self, data, sr, semitones, stretch):
        if semitones == 0 and stretch == 1.0:
            for algo in ["Bungee", "Rubberband", "Pedalboard", "Sox"]:
                self.audio_engine.set_buffer(algo, data, sr)
            return
            
        # We will dispatch threads for each algorithm to prevent blocking the GUI.
        threading.Thread(target=self._render_bungee, args=(data, sr, semitones, stretch)).start()
        threading.Thread(target=self._render_rubberband, args=(data, sr, semitones, stretch)).start()
        threading.Thread(target=self._render_pedalboard, args=(data, sr, semitones, stretch)).start()
        threading.Thread(target=self._render_sox, args=(data, sr, semitones, stretch)).start()

    def _render_bungee(self, data, sr, semitones, stretch):
        t0 = time.perf_counter()
        try:
            from bungee_python import bungee as bungee_lib
            channels = data.shape[1] if len(data.shape) > 1 else 1
            stretcher = bungee_lib.Bungee(sample_rate=sr, channels=channels)
            if semitones != 0:
                stretcher.set_pitch(2.0 ** (semitones / 12.0))
            if stretch != 1.0:
                stretcher.set_speed(stretch)
            audio_in = data.astype(np.float32)
            if len(audio_in.shape) == 1:
                audio_in = audio_in.reshape(-1, 1)
            shifted = np.asarray(stretcher.process(audio_in), dtype=np.float32).copy()
            if channels == 1:
                shifted = shifted.reshape(-1)
            else:
                if shifted.ndim == 1:
                    # Bungee returned mono for stereo input — duplicate channels
                    shifted = np.column_stack([shifted] * channels)
                elif shifted.shape[1] != channels:
                    shifted = np.tile(shifted[:, :1], (1, channels))
            self.audio_engine.set_buffer("Bungee", shifted, sr)
            print(f"Bungee:       {time.perf_counter() - t0:.3f}s")
        except Exception as e:
            print(f"Bungee error: {e}")

    def _render_rubberband(self, data, sr, semitones, stretch):
        t0 = time.perf_counter()
        try:
            import subprocess, tempfile
            formants = self.rb_formants_cb.isChecked()
            r3 = self.rb_engine_combo.currentIndex() == 1
            crisp = self.rb_crisp_spin.value()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                sf.write(f_in.name, data, sr)

                cmd = ["rubberband", "-q"]  # -q = quiet (suppress progress output)
                if r3:
                    cmd += ["--fine"]        # R3 engine: --fine / -3  (NOT --engine 3)
                else:
                    cmd += ["-c", str(crisp)]  # crisp level 0-6, R2 only
                if formants:
                    cmd += ["--formant"]
                if stretch != 1.0:
                    # --time is a duration ratio: > 1 = longer/slower
                    # our slider: stretch=0.5 means half speed → pass 2.0
                    cmd += ["--time", str(1.0 / stretch)]
                if semitones != 0:
                    # --pitch takes semitones directly
                    cmd += ["--pitch", str(semitones)]
                cmd += [f_in.name, f_out.name]

                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0 and r3:
                    # --fine not supported by this rubberband build; retry as R2
                    print("Rubberband: --fine (R3) not supported, falling back to R2")
                    cmd = [c for c in cmd if c != "--fine"]
                    cmd.insert(cmd.index("-q") + 1, "-c")
                    cmd.insert(cmd.index("-c") + 1, str(crisp))
                    result = subprocess.run(cmd, capture_output=True)
                result.check_returncode()
                shifted, out_sr = sf.read(f_out.name)
                self.audio_engine.set_buffer("Rubberband", shifted, sr)

            os.remove(f_in.name)
            os.remove(f_out.name)
            engine_label = "R3" if r3 else f"R2 c={crisp}"
            print(f"Rubberband:   {time.perf_counter() - t0:.3f}s  [{engine_label}{'  formants' if formants else ''}]")
        except Exception as e:
            print(f"Rubberband error: {e}")

    def _render_pedalboard(self, data, sr, semitones, stretch):
        t0 = time.perf_counter()
        try:
            import pedalboard
            data_t = data.T if len(data.shape) > 1 else data.reshape(1, -1)
            data_t = data_t.astype(np.float32)
            formants = self.pb_formants_cb.isChecked()
            transient_mode = self.pb_transient_combo.currentText()
            transient_detector = self.pb_detector_combo.currentText()
            if hasattr(pedalboard, "time_stretch"):
                shifted_t = pedalboard.time_stretch(
                    data_t,
                    float(sr),
                    stretch_factor=stretch,
                    pitch_shift_in_semitones=semitones,
                    high_quality=True,
                    preserve_formants=formants,
                    transient_mode=transient_mode,
                    transient_detector=transient_detector,
                )
            else:
                from pedalboard import Pedalboard, PitchShift
                board = Pedalboard([PitchShift(semitones=semitones)])
                shifted_t = board(data_t, sr)
            shifted = shifted_t.T if len(data.shape) > 1 else shifted_t[0]
            self.audio_engine.set_buffer("Pedalboard", shifted, sr)
            print(f"Pedalboard:   {time.perf_counter() - t0:.3f}s  [{transient_mode}/{transient_detector}{'  formants' if formants else ''}]")
        except Exception as e:
            print(f"Pedalboard error: {e}")

    def _render_sox(self, data, sr, semitones, stretch):
        t0 = time.perf_counter()
        try:
            import subprocess
            import tempfile
            wsola = self.sox_wsola_cb.isChecked()
            audio_type = self.sox_audio_type_combo.currentText()[0].lower()  # 'm', 's', 'l'
            segment = self.sox_segment_spin.value()
            search = self.sox_search_spin.value()
            pitch_quality = self.sox_pitch_quality_combo.currentText()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                sf.write(f_in.name, data, sr)

                # Build sox command directly for full parameter control
                cmd = ["sox", f_in.name, f_out.name]

                if stretch != 1.0:
                    if wsola:
                        # tempo: factor < 1 = slower, > 1 = faster — same convention as our slider
                        cmd += ["tempo", f"-{audio_type}", str(stretch), str(segment), str(search)]
                    else:
                        # stretch: factor < 1 = shorter (faster), so invert our duration ratio
                        cmd += ["stretch", str(1.0 / stretch)]

                if semitones != 0:
                    # pitch only accepts -q (quick cubic) or nothing (high quality band-limited)
                    cents = int(semitones * 100)
                    if pitch_quality == "Quick cubic (-q)":
                        cmd += ["pitch", "-q", str(cents)]
                    else:
                        cmd += ["pitch", str(cents)]

                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    print(f"Sox stderr: {result.stderr.decode(errors='replace').strip()}")
                    result.check_returncode()
                shifted, out_sr = sf.read(f_out.name)
                self.audio_engine.set_buffer("Sox", shifted, out_sr)

            os.remove(f_in.name)
            os.remove(f_out.name)
            mode = f"{'WSOLA/' + audio_type if wsola else 'OLA'}  seg={segment}ms  pitch={'quick' if pitch_quality.startswith('Quick') else 'high'}"
            print(f"Sox:          {time.perf_counter() - t0:.3f}s  [{mode}]")
        except Exception as e:
            print(f"Sox error: {e}")

    def keyPressEvent(self, event):
        # Bind keys 1-9 to algorithms
        if event.key() == Qt.Key.Key_1:
            self.algo_radios["Original"].click()
        elif event.key() == Qt.Key.Key_2:
            self.algo_radios["Bungee"].click()
        elif event.key() == Qt.Key.Key_3:
            self.algo_radios["Rubberband"].click()
        elif event.key() == Qt.Key.Key_4:
            self.algo_radios["Pedalboard"].click()
        elif event.key() == Qt.Key.Key_5:
            self.algo_radios["Sox"].click()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PitchShiftGUI()
    window.show()
    sys.exit(app.exec())
