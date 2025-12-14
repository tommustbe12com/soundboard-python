# soundboard
import sys
import os
import json
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from scipy.signal import resample_poly
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QComboBox, QScrollArea, QFrame,
    QCheckBox, QGroupBox, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
import scipy.signal

# Note to self: maybe want to use librosa later. idk

CONFIG_FILE = "soundboard_config.json"

# Load config or create default
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except Exception:
        config = {}
else:
    config = {}

config.setdefault("sounds", [])
config.setdefault("mic_device", None)
config.setdefault("output_device", None)
config.setdefault("monitor_device", None)
config.setdefault("monitor_enabled", False)

pygame.mixer.init(frequency=44100, size=-16, channels=2)

def stereoize(data):
    """Ensure (N,2) float32 array."""
    if data is None:
        return None
    data = np.asarray(data, dtype="float32")
    if data.ndim == 1:
        return np.column_stack([data, data])
    if data.ndim == 2 and data.shape[1] == 1:
        return np.column_stack([data[:, 0], data[:, 0]])
    # if already (N,2) or (N,>2), then keep first two channels
    if data.ndim == 2 and data.shape[1] >= 2:
        return data[:, :2].astype("float32")
    return data.astype("float32")

# pitch shift for small blocks
def pitch_shift_simple(signal, semitones):
    """
    Real-time pitch shift using resample_poly for audible effect.
    Works with stereo signals (N,2).
    semitones >0 = pitch up, <0 = pitch down
    """
    if abs(semitones) < 1e-3:
        return signal

    factor = 2.0 ** (semitones / 12.0)
    out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        y = signal[:, ch]
        # calculate integer up/down for resample_poly
        up = max(1, int(factor * 100))
        down = 100
        y_shifted = resample_poly(y, up=up, down=down)
        # trim or pad to original length
        if len(y_shifted) > len(y):
            out[:, ch] = y_shifted[:len(y)]
        else:
            out[:len(y_shifted), ch] = y_shifted
    # normalize
    maxv = np.max(np.abs(out))
    if maxv > 1.0:
        out /= maxv
    return out.astype("float32")


# random dsp helpers
def apply_reverb(signal, sr, reverb_ms=120, decay=0.3):
    delay_samples = int(sr * reverb_ms / 1000)
    if delay_samples <= 0 or delay_samples >= len(signal):
        return signal
    out = signal.copy()
    # vectorized loop
    out[delay_samples:] += decay * out[:-delay_samples]
    # clamp
    out = np.clip(out, -1.0, 1.0)
    return out

def apply_distortion(signal, drive=1.0):
    return np.tanh(signal * (1.0 + drive * 4.0)).astype("float32")

def apply_bandpass(signal, sr, low=300, high=3000, order=4):
    nyq = 0.5 * sr
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.99999)
    if low_n >= high_n:
        return signal
    b, a = scipy.signal.butter(order, [low_n, high_n], btype='band')
    out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = scipy.signal.lfilter(b, a, signal[:, ch])
    return out.astype("float32")

def apply_robot(signal, sr, rate=30.0):
    t = np.arange(len(signal)) / float(sr)
    carrier = np.sin(2 * np.pi * rate * t)[:, None]
    mod = signal * carrier
    quant = np.round(mod * 8.0) / 8.0
    return quant.astype("float32")

# GUI APP
class Soundboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TomMustBe12 Soundboard (voice changer)")
        self.setGeometry(120, 120, 780, 760)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.sounds = {}
        self.vol_sliders = {}
        self.sound_rows = {}
        self.lock = threading.Lock()

        # state
        self.active_sounds = []  # list of [sound_array, pos, vol]
        self.fs = 44100
        self.blocksize = 1024
        self.pitch_buffer = np.zeros((self.blocksize * 4, 2), dtype="float32")  # 4 blocks history

        # devices
        self.devices = sd.query_devices()
        self.mic_device = config["mic_device"]
        self.output_device = config["output_device"]
        self.monitor_device = config["monitor_device"]
        self.monitor_enabled = config["monitor_enabled"]

        self.monitor_stream = None
        self.stream = None

        # GUI
        self.init_top_controls()
        self.init_scroll_area(max_height=360)
        self.load_sounds()
        self.add_bottom_buttons()

        # audio start
        self.start_streams()

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.start(250)
        
        self.prev_tail = np.zeros((64,2), dtype="float32")  # same as fade_len


    def init_top_controls(self):
        # device
        devices_box = QGroupBox("Devices")
        devices_layout = QGridLayout()
        devices_box.setLayout(devices_layout)
        devices_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(devices_box)

        self.devices = sd.query_devices()
        mic_names = [d["name"] for d in self.devices if d["max_input_channels"] > 0]
        out_names = [d["name"] for d in self.devices if d["max_output_channels"] > 0]

        devices_layout.addWidget(QLabel("Microphone Device:"), 0, 0)
        self.mic_combo = QComboBox()
        self.mic_combo.addItems(mic_names)
        if self.mic_device in mic_names:
            self.mic_combo.setCurrentText(self.mic_device)
        devices_layout.addWidget(self.mic_combo, 0, 1)

        devices_layout.addWidget(QLabel("Output Device (VB-Cable Input):"), 1, 0)
        self.output_combo = QComboBox()
        self.output_combo.addItems(out_names)
        if self.output_device in out_names:
            self.output_combo.setCurrentText(self.output_device)
        devices_layout.addWidget(self.output_combo, 1, 1)

        self.monitor_checkbox = QCheckBox("Enable Monitoring (Hear Yourself)")
        self.monitor_checkbox.setChecked(self.monitor_enabled)
        self.monitor_checkbox.stateChanged.connect(self.toggle_monitor)
        devices_layout.addWidget(self.monitor_checkbox, 2, 0, 1, 2)

        devices_layout.addWidget(QLabel("Monitor Output Device:"), 3, 0)
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItems(out_names)
        if self.monitor_device in out_names:
            self.monitor_combo.setCurrentText(self.monitor_device)
        devices_layout.addWidget(self.monitor_combo, 3, 1)

        self.mic_combo.currentTextChanged.connect(self.save_config_and_restart_streams)
        self.output_combo.currentTextChanged.connect(self.save_config_and_restart_streams)
        self.monitor_combo.currentTextChanged.connect(self.save_config)

        # voice changer below the device section
        effects_box = QGroupBox("Voice Changer")
        effects_layout = QVBoxLayout()
        effects_box.setLayout(effects_layout)
        self.layout.addWidget(effects_box)

        self.preset_combo = QComboBox()
        presets = [
            "Off", "Pitch Shift (deep/chipmunk)", "Robot / Metallic",
            "Radio / Walkie", "Megaphone", "Reverb", "Demon", "Custom"
        ]
        self.preset_combo.addItems(presets)
        self.preset_combo.currentTextChanged.connect(self.on_preset_change)
        effects_layout.addWidget(self.preset_combo)

        custom_grid = QGridLayout()
        r = 0

        def add_slider(label, slider):
            nonlocal r
            custom_grid.addWidget(QLabel(label), r, 0)
            custom_grid.addWidget(slider, r, 1)
            r += 1

        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-24, 24)
        add_slider("Pitch (semitones)", self.pitch_slider)

        self.reverb_slider = QSlider(Qt.Orientation.Horizontal)
        self.reverb_slider.setRange(0, 500)
        add_slider("Reverb (ms)", self.reverb_slider)

        self.dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.dist_slider.setRange(0, 100)
        add_slider("Distortion", self.dist_slider)

        self.robot_slider = QSlider(Qt.Orientation.Horizontal)
        self.robot_slider.setRange(10, 200)
        self.robot_slider.setValue(30)
        add_slider("Robot Rate (Hz)", self.robot_slider)

        self.band_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.band_low_slider.setRange(50, 1000)
        add_slider("Band Low (Hz)", self.band_low_slider)

        self.band_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.band_high_slider.setRange(1000, 12000)
        add_slider("Band High (Hz)", self.band_high_slider)

        effects_layout.addLayout(custom_grid)

        btn_row = QHBoxLayout()
        self.effect_toggle = QPushButton("Enable Effect")
        self.effect_toggle.setCheckable(True)
        self.effect_toggle.clicked.connect(self.toggle_effects)
        btn_row.addWidget(self.effect_toggle)

        for q in ["Deep", "Chipmunk", "Robot", "Radio", "Megaphone", "Demon"]:
            b = QPushButton(q)
            b.clicked.connect(lambda _, x=q: self.apply_quick_preset(x))
            btn_row.addWidget(b)

        effects_layout.addLayout(btn_row)

        self.test_btn = QPushButton("Play Test Voice (1s)")
        self.test_btn.clicked.connect(self.record_and_play_test)
        effects_layout.addWidget(self.test_btn)


    def init_scroll_area(self, max_height=360):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_frame = QFrame()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setSpacing(6)
        self.scroll_layout.setContentsMargins(6, 6, 6, 6)
        self.scroll_frame.setLayout(self.scroll_layout)
        scroll.setWidget(self.scroll_frame)
        scroll.setMaximumHeight(max_height)
        self.layout.addWidget(scroll)

    def add_bottom_buttons(self):
        row = QHBoxLayout()
        add_btn = QPushButton("Add New Sound")
        add_btn.clicked.connect(self.add_sound)
        row.addWidget(add_btn)

        stop_btn = QPushButton("STOP ALL SOUNDS")
        stop_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        stop_btn.clicked.connect(self.stop_all_sounds)
        row.addWidget(stop_btn)

        self.layout.addLayout(row)

    def load_sounds(self):
        for sound_file in config["sounds"]:
            if os.path.exists(sound_file):
                data, sr = sf.read(sound_file, dtype="float32")
                if sr != self.fs:
                    # stack overflow sourced bruh
                    if data.ndim == 1:
                        data = np.interp(
                            np.linspace(0, len(data) - 1, int(len(data) * self.fs / sr)),
                            np.arange(len(data)), data
                        )
                    else:
                        data = np.vstack([
                            np.interp(np.linspace(0, data.shape[0] - 1, int(data.shape[0] * self.fs / sr)),
                                      np.arange(data.shape[0]), data[:, ch])
                            for ch in range(min(2, data.shape[1]))
                        ]).T
                data = stereoize(data)
                self.sounds[sound_file] = data
                self.add_sound_row(sound_file)

    def add_sound_row(self, sound_file):
        frame = QFrame()
        layout = QHBoxLayout()
        frame.setLayout(layout)

        play_btn = QPushButton(os.path.basename(sound_file))
        play_btn.clicked.connect(lambda _, f=sound_file: self.trigger_sound(f))
        layout.addWidget(play_btn)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(100)
        layout.addWidget(slider)
        self.vol_sliders[sound_file] = slider

        trash_btn = QPushButton("🗑️")
        trash_btn.setToolTip("Remove sound")
        trash_btn.clicked.connect(lambda _, f=sound_file, fr=frame: self.remove_sound(f, fr))
        layout.addWidget(trash_btn)

        self.scroll_layout.addWidget(frame)
        self.sound_rows[sound_file] = frame

    def add_sound(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Sound", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")
        if file_path:
            data, sr = sf.read(file_path, dtype="float32")
            if sr != self.fs:
                if data.ndim == 1:
                    data = np.interp(
                        np.linspace(0, len(data) - 1, int(len(data) * self.fs / sr)),
                        np.arange(len(data)), data
                    )
                else:
                    data = np.vstack([
                        np.interp(np.linspace(0, data.shape[0] - 1, int(data.shape[0] * self.fs / sr)),
                                  np.arange(data.shape[0]), data[:, ch])
                        for ch in range(min(2, data.shape[1]))
                    ]).T
            data = stereoize(data)
            self.sounds[file_path] = data
            self.add_sound_row(file_path)
            self.save_config()

    def remove_sound(self, path, frame_widget):
        if path in self.sounds:
            if frame_widget:
                frame_widget.setParent(None)
            with self.lock:
                if path in self.sounds:
                    del self.sounds[path]
            if path in self.vol_sliders:
                del self.vol_sliders[path]
            if path in config["sounds"]:
                config["sounds"].remove(path)
            self.save_config()

    def stop_all_sounds(self):
        with self.lock:
            self.active_sounds.clear()

    def trigger_sound(self, sound_file):
        vol = self.vol_sliders[sound_file].value() / 100.0
        with self.lock:
            self.active_sounds.append([self.sounds[sound_file], 0, vol])

    # CONFIG
    def save_config(self):
        config["mic_device"] = self.mic_combo.currentText()
        config["output_device"] = self.output_combo.currentText()
        config["monitor_device"] = self.monitor_combo.currentText()
        config["monitor_enabled"] = self.monitor_enabled
        config["sounds"] = list(self.sounds.keys())
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except Exception:
            pass

    def save_config_and_restart_streams(self, *_):
        self.save_config()
        self.restart_streams()

    # EFFECTS
    def on_preset_change(self, text):
        if text == "Pitch Shift (deep/chipmunk)":
            self.pitch_slider.setValue(-8); self.reverb_slider.setValue(0); self.dist_slider.setValue(0)
        elif text == "Robot / Metallic":
            self.pitch_slider.setValue(0); self.reverb_slider.setValue(0); self.robot_slider.setValue(75)
        elif text == "Radio / Walkie":
            self.band_low_slider.setValue(300); self.band_high_slider.setValue(3000); self.dist_slider.setValue(8)
        elif text == "Megaphone":
            self.band_low_slider.setValue(500); self.band_high_slider.setValue(4000); self.dist_slider.setValue(30)
        elif text == "Reverb":
            self.reverb_slider.setValue(180)
        elif text == "Demon":
            self.pitch_slider.setValue(-10); self.dist_slider.setValue(40); self.reverb_slider.setValue(140)
        # no direct save; sliders live-read in processing

    def apply_quick_preset(self, name):
        if name == "Deep":
            self.preset_combo.setCurrentText("Pitch Shift (deep/chipmunk)"); self.pitch_slider.setValue(-12)
        elif name == "Chipmunk":
            self.preset_combo.setCurrentText("Pitch Shift (deep/chipmunk)"); self.pitch_slider.setValue(12)
        elif name == "Robot":
            self.preset_combo.setCurrentText("Robot / Metallic"); self.robot_slider.setValue(60)
        elif name == "Radio":
            self.preset_combo.setCurrentText("Radio / Walkie"); self.band_low_slider.setValue(300); self.band_high_slider.setValue(3000)
        elif name == "Megaphone":
            self.preset_combo.setCurrentText("Megaphone"); self.dist_slider.setValue(40)
        elif name == "Demon":
            self.preset_combo.setCurrentText("Demon"); self.pitch_slider.setValue(-16); self.dist_slider.setValue(60); self.reverb_slider.setValue(220)
        self.effect_toggle.setChecked(True)
        self.effect_toggle.setText("Disable Effect")

    def toggle_effects(self):
        active = self.effect_toggle.isChecked()
        self.effect_toggle.setText("Disable Effect" if active else "Enable Effect")

    # test voice help
    def record_and_play_test(self):
        """Record 1s from selected mic, process it and queue to active_sounds so it goes through same pipeline."""
        mic_name = self.mic_combo.currentText()
        mic_idx = None
        for i, d in enumerate(self.devices):
            if d["name"] == mic_name and d["max_input_channels"] > 0:
                mic_idx = i
                break
        if mic_idx is None:
            print("No mic selected for test.")
            return
        duration = 1.0
        try:
            rec = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=2, dtype="float32", device=mic_idx)
            sd.wait()
            rec = stereoize(rec)
            processed = self.apply_effects_to_buffer(rec)
            with self.lock:
                self.active_sounds.append([processed, 0, 1.0])
        except Exception as e:
            print("Test record failed:", e)

    def apply_effects_to_buffer(self, buffer):
        """Process a whole buffer with same logic used in callback (used by test button)."""
        processed = buffer.copy()
        if not self.effect_toggle.isChecked() or self.preset_combo.currentText() == "Off":
            return processed
        preset = self.preset_combo.currentText()
        pitch = float(self.pitch_slider.value())
        reverb_ms = int(self.reverb_slider.value())
        distortion = float(self.dist_slider.value()) / 100.0
        robot_rate = float(self.robot_slider.value())
        band_low = float(self.band_low_slider.value())
        band_high = float(self.band_high_slider.value())

        try:
            if preset == "Pitch Shift (deep/chipmunk)":
                self.pitch_buffer = np.vstack([self.pitch_buffer, mic])[-self.blocksize*4:]

                shifted = pitch_shift_simple(self.pitch_buffer, pitch)

                processed = shifted[-frames:]

                # crossfade didn't work to block crackles.
                fade_len = min(64, len(processed))  # adjust length???
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                processed[:fade_len] *= fade_in[:, None]
                processed[:fade_len] += self.prev_tail[:fade_len] * fade_out[:, None]

                # save tail for next
                self.prev_tail = processed[-fade_len:].copy()


            elif preset == "Robot / Metallic":
                p = pitch_shift_simple(processed, 0)
                processed = apply_robot(p, self.fs, rate=robot_rate)
                processed = apply_distortion(processed, drive=distortion * 0.6)
            elif preset == "Radio / Walkie":
                processed = apply_bandpass(processed, self.fs, low=300, high=3000)
                processed = apply_distortion(processed, drive=0.08)
            elif preset == "Megaphone":
                processed = apply_bandpass(processed, self.fs, low=500, high=4000)
                processed = apply_distortion(processed, drive=0.3)
            elif preset == "Reverb":
                processed = apply_reverb(processed, self.fs, reverb_ms=max(40, reverb_ms), decay=0.28)
            elif preset == "Demon":
                processed = pitch_shift_simple(processed, -10)
                processed = apply_distortion(processed, drive=0.5)
                processed = apply_reverb(processed, self.fs, reverb_ms=160, decay=0.32)
            elif preset == "Custom":
                if abs(pitch) > 0.001:
                    processed = pitch_shift_simple(processed, pitch)
                processed = apply_bandpass(processed, self.fs, low=band_low, high=band_high)
                if robot_rate > 0:
                    processed = apply_robot(processed, self.fs, rate=robot_rate) * (1.0 - distortion)
                if distortion > 0.001:
                    processed = apply_distortion(processed, drive=distortion)
                if reverb_ms > 2:
                    processed = apply_reverb(processed, self.fs, reverb_ms=int(reverb_ms), decay=0.25)
        except Exception as e:
            print("Processing error in test:", e)
            processed = buffer
        # ensure correct shape and dtype
        processed = stereoize(processed)
        if len(processed) < len(buffer):
            pad = np.zeros((len(buffer) - len(processed), 2), dtype="float32")
            processed = np.vstack([processed, pad])
        elif len(processed) > len(buffer):
            processed = processed[:len(buffer)]
        return processed.astype("float32")

    # AUDIO streams
    def start_monitor_stream(self):
        if self.monitor_stream:
            try:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None

        monitor_name = self.monitor_combo.currentText()
        monitor_idx = None
        for i, d in enumerate(self.devices):
            if d["name"] == monitor_name and d["max_output_channels"] > 0:
                monitor_idx = i
                break
        if monitor_idx is None:
            return

        self.monitor_stream = sd.OutputStream(
            samplerate=self.fs, blocksize=self.blocksize,
            device=monitor_idx, channels=2, dtype="float32"
        )
        self.monitor_stream.start()

    def start_streams(self):
        # refresh
        self.devices = sd.query_devices()
        mic_name = self.mic_combo.currentText()
        out_name = self.output_combo.currentText()

        mic_idx = None
        out_idx = None
        for i, d in enumerate(self.devices):
            if mic_idx is None and d["name"] == mic_name and d["max_input_channels"] > 0:
                mic_idx = i
            if out_idx is None and d["name"] == out_name and d["max_output_channels"] > 0:
                out_idx = i

        if mic_idx is None or out_idx is None:
            print("Error: microphone or output device not found. Check device selection.")
            return

        # monitor 
        if self.monitor_checkbox.isChecked():
            self.start_monitor_stream()

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        def callback(indata, outdata, frames, time, status):
            if status:
                # print status 
                print("Stream status:", status)

            # safe mic copy
            try:
                mic = stereoize(np.copy(indata))
            except Exception:
                mic = np.zeros((frames, 2), dtype="float32")

            # read sliders livve
            preset = self.preset_combo.currentText()
            pitch = float(self.pitch_slider.value())
            reverb_ms = int(self.reverb_slider.value())
            distortion = float(self.dist_slider.value()) / 100.0
            robot_rate = float(self.robot_slider.value())
            band_low = float(self.band_low_slider.value())
            band_high = float(self.band_high_slider.value())

            processed = mic
            if self.effect_toggle.isChecked() and preset != "Off":
                try:
                    if preset == "Pitch Shift (deep/chipmunk)":
                        processed = pitch_shift_simple(processed, pitch)
                    elif preset == "Robot / Metallic":
                        p = pitch_shift_simple(processed, 0)
                        processed = apply_robot(p, self.fs, rate=robot_rate)
                        processed = apply_distortion(processed, drive=distortion * 0.6)
                    elif preset == "Radio / Walkie":
                        processed = apply_bandpass(processed, self.fs, low=300, high=3000)
                        processed = apply_distortion(processed, drive=0.08)
                    elif preset == "Megaphone":
                        processed = apply_bandpass(processed, self.fs, low=500, high=4000)
                        processed = apply_distortion(processed, drive=0.3)
                    elif preset == "Reverb":
                        processed = apply_reverb(processed, self.fs, reverb_ms=max(40, reverb_ms), decay=0.28)
                    elif preset == "Demon":
                        processed = pitch_shift_simple(processed, -10)
                        processed = apply_distortion(processed, drive=0.5)
                        processed = apply_reverb(processed, self.fs, reverb_ms=160, decay=0.32)
                    elif preset == "Custom":
                        if abs(pitch) > 0.001:
                            processed = pitch_shift_simple(processed, pitch)
                        processed = apply_bandpass(processed, self.fs, low=band_low, high=band_high)
                        if robot_rate > 0:
                            processed = apply_robot(processed, self.fs, rate=robot_rate) * (1.0 - distortion)
                        if distortion > 0.001:
                            processed = apply_distortion(processed, drive=distortion)
                        if reverb_ms > 2:
                            processed = apply_reverb(processed, self.fs, reverb_ms=int(reverb_ms), decay=0.25)
                except Exception as e:
                    print("Effect processing error:", e)
                    processed = mic

            # start output with processed mic
            output = processed.copy()

            # mix queued
            with self.lock:
                new_active = []
                for sound_array, pos, vol in self.active_sounds:
                    sa = sound_array
                    chunk = sa[pos:pos + frames]
                    l = min(len(chunk), frames)
                    if l > 0:
                        output[:l] += chunk[:l] * vol
                    if pos + frames < len(sa):
                        new_active.append([sa, pos + frames, vol])
                self.active_sounds = new_active

            # clip output
            output = np.clip(output, -1.0, 1.0)
            try:
                outdata[:] = output
            except Exception as e:
                outdata[:] = np.zeros_like(outdata)
                print("Outdata write error:", e)

            # write to monitor local
            if self.monitor_checkbox.isChecked():
                try:
                    if self.monitor_stream:
                        self.monitor_stream.write(output.copy())
                except Exception:
                    pass

        # create stream with explicit device tuple
        try:
            self.stream = sd.Stream(
                samplerate=self.fs,
                blocksize=self.blocksize,
                device=(mic_idx, out_idx),
                channels=2,
                dtype="float32",
                callback=callback
            )
            self.stream.start()
        except Exception as e:
            print("Failed to start main stream:", e)

    def restart_streams(self):
        try:
            if self.stream:
                self.stream.stop(); self.stream.close(); self.stream = None
        except Exception:
            pass
        try:
            if self.monitor_stream:
                self.monitor_stream.stop(); self.monitor_stream.close(); self.monitor_stream = None
        except Exception:
            pass
        self.devices = sd.query_devices()
        self.start_streams()

    def toggle_monitor(self):
        self.monitor_enabled = self.monitor_checkbox.isChecked()
        self.save_config()
        if self.monitor_enabled:
            self.start_monitor_stream()
        else:
            try:
                if self.monitor_stream:
                    self.monitor_stream.stop()
                    self.monitor_stream.close()
            except Exception:
                pass
            self.monitor_stream = None

    def closeEvent(self, event):
        try:
            if self.stream:
                self.stream.stop(); self.stream.close()
        except Exception:
            pass
        try:
            if self.monitor_stream:
                self.monitor_stream.stop(); self.monitor_stream.close()
        except Exception:
            pass
        self.save_config()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = Soundboard()
    window.show()
    sys.exit(app.exec())
