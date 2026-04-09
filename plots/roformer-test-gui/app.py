import sys
import os
import glob
import json
import threading
import time
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSlider, QPushButton, QLabel, QRadioButton, QButtonGroup,
    QGroupBox, QFormLayout, QGridLayout, QMessageBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

import numpy as np
import sounddevice as sd
import soundfile as sf

# Update paths to point to project directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = "/run/media/kim/Mantu/ai-music/Goa_Separated_crops"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "bs-roformer")
SEPARATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "separated_tracks")

from separator import separate_file_to_dir, separate_file_with_demucs, get_model_stems, is_model_supported

DEMUCS_MODELS = ["demucs/htdemucs", "demucs/htdemucs_ft"]
DEMUCS_STEMS = ["drums", "bass", "other", "vocals"]

class AudioEngine(QObject):
    playback_position_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.stream = None
        self.is_playing = False
        self.playhead = 0
        self.samplerate = 44100
        
        # buffers[model_name][stem_name] = data
        self.buffers = {}
        
        self.active_model = None
        self.active_stem = None
        
        self._pending_model = None
        self._pending_stem = None
        self._crossfade_frames = 441  # 10ms crossfade at 44.1kHz

        self.current_buffer = None
        self.fade_buffer = None
        self.fade_samples_remaining = 0

    def load_buffers(self, all_buffers_dict, samplerate):
        self.pause()
        self.buffers = all_buffers_dict
        self.samplerate = samplerate
        self.playhead = 0
        self.current_buffer = None
        self.fade_buffer = None
        self.fade_samples_remaining = 0
        
        # Default activate first model and first stem if available
        if self.buffers:
            models = list(self.buffers.keys())
            if models:
                self.active_model = models[0]
                stems = list(self.buffers[self.active_model].keys())
                if stems:
                    self.active_stem = stems[0]
                    self.current_buffer = self.buffers[self.active_model][self.active_stem]

    def switch_track(self, model_name, stem_name):
        if not self.buffers: return
        
        if model_name not in self.buffers:
            model_name = list(self.buffers.keys())[0] if self.buffers else None
            
        if stem_name not in self.buffers.get(model_name, {}):
            stems = list(self.buffers.get(model_name, {}).keys())
            stem_name = stems[0] if stems else None
            
        if model_name == self.active_model and stem_name == self.active_stem:
            return

        if self.stream is not None and self.stream.active:
            self._pending_model = model_name
            self._pending_stem = stem_name
        else:
            self.active_model = model_name
            self.active_stem = stem_name
            if model_name and stem_name:
                self.current_buffer = self.buffers[model_name][stem_name]

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        # Handle pending switch with crossfade
        if self._pending_model is not None and self._pending_stem is not None:
            new_buf = self.buffers.get(self._pending_model, {}).get(self._pending_stem)
            if new_buf is not None and self.current_buffer is not None:
                self.fade_buffer = self.current_buffer
                self.current_buffer = new_buf
                self.fade_samples_remaining = self._crossfade_frames
            elif new_buf is not None:
                self.current_buffer = new_buf
                
            self.active_model = self._pending_model
            self.active_stem = self._pending_stem
            self._pending_model = None
            self._pending_stem = None

        if not self.is_playing or self.current_buffer is None or len(self.current_buffer) == 0:
            outdata[:] = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
            return

        buffer_len = len(self.current_buffer)
        if self.playhead >= buffer_len:
            self.playhead = 0

        remaining = frames
        out_idx = 0
        while remaining > 0:
            chunk = min(remaining, buffer_len - self.playhead)
            out_chunk = self.current_buffer[self.playhead:self.playhead+chunk].copy()
            
            # Apply crossfade if active
            if self.fade_samples_remaining > 0 and self.fade_buffer is not None:
                fade_chunk = min(chunk, self.fade_samples_remaining)
                
                # Calculate linear fade weights
                start_w = self.fade_samples_remaining / self._crossfade_frames
                end_w = (self.fade_samples_remaining - fade_chunk) / self._crossfade_frames
                
                weights_old = np.linspace(start_w, end_w, fade_chunk, dtype=np.float32)
                weights_new = 1.0 - weights_old
                
                if out_chunk.ndim > 1:
                    weights_old = weights_old[:, np.newaxis]
                    weights_new = weights_new[:, np.newaxis]
                
                old_chunk = self.fade_buffer[self.playhead:self.playhead+fade_chunk]
                
                # Pad old_chunk with zeros if it's shorter than the current playhead
                if len(old_chunk) < fade_chunk:
                    pad_len = fade_chunk - len(old_chunk)
                    if old_chunk.ndim > 1:
                        old_chunk = np.pad(old_chunk, ((0, pad_len), (0, 0)), mode='constant')
                    else:
                        old_chunk = np.pad(old_chunk, (0, pad_len), mode='constant')

                out_chunk[:fade_chunk] = (old_chunk * weights_old) + (out_chunk[:fade_chunk] * weights_new)
                self.fade_samples_remaining -= fade_chunk
                
                if self.fade_samples_remaining <= 0:
                    self.fade_buffer = None

            outdata[out_idx:out_idx+chunk] = out_chunk
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

class WorkerThread(threading.Thread):
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args, daemon=True)

class RoformerTestGUI(QMainWindow):
    # Signals to update UI safely from non-GUI threads
    separation_progress = pyqtSignal(str)
    separation_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BS-Roformer Test GUI")
        self.resize(1000, 600)
        
        self.audio_engine = AudioEngine()
        self.tracks = self._scan_dataset()
        self.models = self._scan_models() + DEMUCS_MODELS
        self.model_stems = {m: get_model_stems(MODELS_DIR, m) for m in self.models if not m.startswith("demucs/")}
        for dm in DEMUCS_MODELS:
            self.model_stems[dm] = DEMUCS_STEMS
        self.separated_data_cache = {} # {crop_name: {model: {stem: np_array}}}
        
        # Connect signals
        self.separation_progress.connect(self.update_status)
        self.separation_finished.connect(self.on_separation_finished)
        
        self.init_ui()

    def _scan_dataset(self):
        """Scans DATASET_DIR for tracks."""
        tracks = {}
        if not os.path.exists(DATASET_DIR):
            return tracks
            
        for d in os.listdir(DATASET_DIR):
            full_path = os.path.join(DATASET_DIR, d)
            if os.path.isdir(full_path) and glob.glob(os.path.join(full_path, "*.flac")):
                tracks[d] = full_path
        return tracks

    def _scan_models(self):
        """Scans MODELS_DIR for BS-Roformer models, expanding subfolder models."""
        models = []
        if not os.path.exists(MODELS_DIR):
            return models

        def has_checkpoint(folder):
            return any(
                glob.glob(os.path.join(folder, f'*{ext}'))
                for ext in ('*.ckpt', '*.pth', '*.safetensors')
            )

        for d in sorted(os.listdir(MODELS_DIR)):
            folder = os.path.join(MODELS_DIR, d)
            if not os.path.isdir(folder):
                continue
            if has_checkpoint(folder):
                if is_model_supported(MODELS_DIR, d):
                    models.append(d)
                else:
                    print(f"[model scan] Skipping {d}: MelBandRoformer not supported")
            else:
                # Expand subfolders (e.g. HyperACE/v1-inst)
                subdirs = sorted(
                    s for s in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, s)) and has_checkpoint(os.path.join(folder, s))
                )
                for s in subdirs:
                    name = f"{d}/{s}"
                    if is_model_supported(MODELS_DIR, name):
                        models.append(name)
                    else:
                        print(f"[model scan] Skipping {name}: MelBandRoformer not supported")

        return models

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- 1. Track & Crop Selection ---
        track_group = QGroupBox("Target Crop Selection")
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
        
        track_group.setLayout(track_layout)
        main_layout.addWidget(track_group)

        # --- 2. Action Area ---
        action_layout = QHBoxLayout()
        
        self.sep_btn = QPushButton("Run Separation on All Models")
        self.sep_btn.clicked.connect(self.on_run_separation)
        self.sep_btn.setMinimumHeight(40)
        action_layout.addWidget(self.sep_btn)
        
        self.status_label = QLabel("Idle")
        action_layout.addWidget(self.status_label, 1)
        
        main_layout.addLayout(action_layout)

        # --- 3. Demucs Parameters ---
        demucs_group = QGroupBox("Demucs Parameters")
        demucs_layout = QHBoxLayout()

        self.demucs_model_group = QButtonGroup(self)
        for i, dm in enumerate(DEMUCS_MODELS):
            label = dm.split("/")[1]  # "htdemucs" or "htdemucs_ft"
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            self.demucs_model_group.addButton(rb, i)
            demucs_layout.addWidget(rb)

        demucs_layout.addSpacing(20)
        demucs_layout.addWidget(QLabel("Shifts:"))
        self.demucs_shifts = QSpinBox()
        self.demucs_shifts.setRange(0, 10)
        self.demucs_shifts.setValue(0)
        self.demucs_shifts.setFixedWidth(60)
        demucs_layout.addWidget(self.demucs_shifts)

        demucs_layout.addSpacing(20)
        demucs_layout.addWidget(QLabel("Overlap:"))
        self.demucs_overlap = QDoubleSpinBox()
        self.demucs_overlap.setRange(0.05, 0.50)
        self.demucs_overlap.setSingleStep(0.05)
        self.demucs_overlap.setValue(0.25)
        self.demucs_overlap.setDecimals(2)
        self.demucs_overlap.setFixedWidth(70)
        demucs_layout.addWidget(self.demucs_overlap)

        demucs_layout.addStretch()
        demucs_group.setLayout(demucs_layout)
        main_layout.addWidget(demucs_group)

        # --- 4. Matrix (Models vs Stems) ---
        matrix_layout = QHBoxLayout()
        
        # Models Box
        model_group = QGroupBox("Model Override")
        model_inner = QVBoxLayout()
        self.model_btn_group = QButtonGroup(self)
        self.model_radios = {}
        for i, model in enumerate(self.models):
            rb = QRadioButton(model)
            rb.setProperty("model_name", model)
            if i == 0: rb.setChecked(True)
            model_inner.addWidget(rb)
            self.model_btn_group.addButton(rb, i)
            self.model_radios[model] = rb
        
        if not self.models:
            model_inner.addWidget(QLabel("No models found."))
            
        model_inner.addStretch()
        model_group.setLayout(model_inner)
        self.model_btn_group.idClicked.connect(self.on_matrix_changed)
        matrix_layout.addWidget(model_group, 1)

        # Stems Box
        stem_group = QGroupBox("Stem Isolation")
        self.stem_inner_layout = QVBoxLayout()
        self.stem_btn_group = QButtonGroup(self)
        self.stem_radios = {}
        stem_group.setLayout(self.stem_inner_layout)
        matrix_layout.addWidget(stem_group, 1)

        # Populate stems for the initially selected model
        initial_model = self.models[0] if self.models else None
        self._rebuild_stem_buttons(self.model_stems.get(initial_model, []))
        
        main_layout.addLayout(matrix_layout)

        # --- 5. Transport ---
        transport_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setMinimumHeight(40)
        self.play_btn.clicked.connect(self.toggle_playback)
        transport_layout.addWidget(self.play_btn)
        
        self.pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_slider.setMinimum(0)
        self.pos_slider.setMaximum(1000)
        self.pos_slider.sliderReleased.connect(self.on_pos_released)
        transport_layout.addWidget(self.pos_slider)
        
        main_layout.addLayout(transport_layout)
        
        # Initial trigger
        if self.tracks:
            self.on_track_changed(self.track_combo.currentText())

    def _rebuild_stem_buttons(self, stems):
        while self.stem_inner_layout.count():
            item = self.stem_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.stem_btn_group = QButtonGroup(self)
        self.stem_radios = {}
        for i, stem in enumerate(stems):
            rb = QRadioButton(stem.capitalize())
            rb.setProperty("stem_name", stem)
            if i == 0:
                rb.setChecked(True)
            self.stem_inner_layout.addWidget(rb)
            self.stem_btn_group.addButton(rb, i)
            self.stem_radios[stem] = rb
        self.stem_inner_layout.addStretch()
        self.stem_btn_group.idClicked.connect(self.on_matrix_changed)
        # Sync audio engine to first stem of new model
        if stems:
            active_model = next((m for m, rb in self.model_radios.items() if rb.isChecked()), None)
            if active_model:
                self.audio_engine.switch_track(active_model, stems[0])

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
                    self.crop_positions.append(idx)
        
        self.crop_positions.sort()
        
        self.crop_combo.blockSignals(True)
        self.crop_combo.clear()
        self.crop_combo.addItems([str(c) for c in self.crop_positions])
        self.crop_combo.blockSignals(False)
        
        if self.crop_positions:
            self.on_crop_changed(str(self.crop_positions[0]))

    def get_current_audio_file(self):
        track_path = self.tracks[self.track_combo.currentText()]
        track_name = os.path.basename(track_path)
        crop_idx = self.crop_combo.currentText()
        crop_base = f"{track_name}_{crop_idx}"
        for ext in (".flac", ".ogg", ".wav", ".mp3"):
            audio_file = os.path.join(track_path, f"{crop_base}{ext}")
            if os.path.exists(audio_file):
                return audio_file, crop_base
        # Return flac path as fallback (will trigger "not found" error with correct name)
        return os.path.join(track_path, f"{crop_base}.flac"), crop_base

    def on_crop_changed(self, crop_idx_str):
        if not crop_idx_str: return
        
        audio_file, crop_base = self.get_current_audio_file()
        
        # Check if already separated
        crop_sep_dir = os.path.join(SEPARATED_DIR, crop_base)
        has_separations = False
        
        if os.path.exists(crop_sep_dir):
            model_dirs = os.listdir(crop_sep_dir)
            if model_dirs:
                has_separations = True
        
        if has_separations:
            self.load_stems_to_memory(crop_base)
            self.sep_btn.setText("Separations Loaded ✅")
            self.sep_btn.setEnabled(False)
        else:
            self.audio_engine.pause()
            self.sep_btn.setText("Run Separation on All Models")
            self.sep_btn.setEnabled(True)
            self.update_status("Crop selected. Awaiting separation.")

    def update_status(self, msg):
        self.status_label.setText(msg)

    def on_run_separation(self):
        if not self.models:
            QMessageBox.warning(self, "No Models", "No BS-Roformer models found in models directory.")
            return

        audio_file, crop_base = self.get_current_audio_file()
        if not os.path.exists(audio_file):
            QMessageBox.warning(self, "Not found", f"Audio file {audio_file} not found.")
            return

        self.sep_btn.setEnabled(False)
        self.status_label.setText("Preparing...")
        
        # Launch background thread
        WorkerThread(target=self._run_separation_job, args=(audio_file, crop_base)).start()

    def _run_separation_job(self, input_file, crop_base):
        demucs_shifts = self.demucs_shifts.value()
        demucs_overlap = self.demucs_overlap.value()
        # Determine which Demucs model to run based on current radio selection
        demucs_model_idx = self.demucs_model_group.checkedId()
        active_demucs_model = DEMUCS_MODELS[demucs_model_idx] if demucs_model_idx >= 0 else DEMUCS_MODELS[0]

        for i, model in enumerate(self.models):
            self.separation_progress.emit(f"Separating ({i+1}/{len(self.models)}): {model}")
            output_dir = os.path.join(SEPARATED_DIR, crop_base, model)

            try:
                if model.startswith("demucs/"):
                    if model != active_demucs_model:
                        continue  # Only run the selected Demucs variant
                    demucs_name = model.split("/")[1]
                    separate_file_with_demucs(
                        input_file=Path(input_file),
                        output_dir=Path(output_dir),
                        model_name=demucs_name,
                        shifts=demucs_shifts,
                        overlap=demucs_overlap,
                        device='cuda',
                        callback=lambda msg: self.separation_progress.emit(msg),
                    )
                else:
                    separate_file_to_dir(
                        input_file=Path(input_file),
                        output_dir=Path(output_dir),
                        model_name=model,
                        model_dir=MODELS_DIR,
                        batch_size=1,
                        device='cuda'
                    )
            except Exception as e:
                self.separation_progress.emit(f"Error on {model}: {e}")
                print(f"Error on {model}: {e}")

        self.separation_finished.emit()

    def on_separation_finished(self):
        self.update_status("Separation complete!")
        self.sep_btn.setText("Separations Loaded ✅")
        
        _, crop_base = self.get_current_audio_file()
        self.load_stems_to_memory(crop_base)

    def load_stems_to_memory(self, crop_base):
        crop_sep_dir = os.path.join(SEPARATED_DIR, crop_base)
        self.update_status(f"Loading files into RAM...")
        QApplication.processEvents()
        
        all_buffers = {}
        max_sr = 44100
        
        for model in self.models:
            model_dir = os.path.join(crop_sep_dir, model)
            if not os.path.exists(model_dir):
                continue
                
            model_buffers = {}
            for flac in glob.glob(os.path.join(model_dir, "*.flac")):
                stem_name = os.path.basename(flac).replace(".flac", "")
                data, sr = sf.read(flac)
                if data.ndim == 1:
                    data = np.stack([data, data], axis=-1)
                model_buffers[stem_name] = data
                max_sr = sr
                
            if model_buffers:
                all_buffers[model] = model_buffers
                
        self.audio_engine.load_buffers(all_buffers, max_sr)
        self.update_status(f"Loaded {len(all_buffers)} model outputs.")
        self.on_matrix_changed() # Force setup of active buffers

    def on_matrix_changed(self):
        active_model = None
        for model, rb in self.model_radios.items():
            if rb.isChecked():
                active_model = model
                break

        # Rebuild stems when the model changes
        if active_model and active_model != getattr(self, '_last_active_model', None):
            self._last_active_model = active_model
            self._rebuild_stem_buttons(self.model_stems.get(active_model, []))
            return  # _rebuild_stem_buttons handles the audio switch

        active_stem = None
        for stem, rb in self.stem_radios.items():
            if rb.isChecked():
                active_stem = stem
                break

        if active_model and active_stem:
            self.audio_engine.switch_track(active_model, active_stem)

    def on_pos_released(self):
        val = self.pos_slider.value() / 1000.0
        if self.audio_engine.current_buffer is not None:
            new_pos = int(val * len(self.audio_engine.current_buffer))
            self.audio_engine.playhead = min(new_pos, len(self.audio_engine.current_buffer) - 1)

    def toggle_playback(self):
        if self.audio_engine.is_playing:
            self.audio_engine.pause()
            self.play_btn.setText("Play")
        else:
            if not self.audio_engine.buffers:
                return # Can't play if no separations
            self.audio_engine.start()
            self.play_btn.setText("Pause")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RoformerTestGUI()
    window.show()
    sys.exit(app.exec())
