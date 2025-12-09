import sys, os, json
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QComboBox, QScrollArea, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt

CONFIG_FILE = "soundboard_config.json"

# load da config if it exists or create
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except:
        config = {}
else:
    config = {}

# Do all the keys exist?
config.setdefault("sounds", [])
config.setdefault("mic_device", None)
config.setdefault("output_device", None)
config.setdefault("monitor_device", None)
config.setdefault("monitor_enabled", False)

pygame.mixer.init(frequency=44100, size=-16, channels=2)

def stereoize(data):
    if len(data.shape) == 1:
        return np.column_stack([data, data])
    if data.shape[1] == 1:
        return np.column_stack([data[:, 0], data[:, 0]])
    return data

class Soundboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TomMustBe12 Soundboard")
        self.setGeometry(200, 200, 600, 650)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.sounds = {}
        self.vol_sliders = {}

        self.mic_device = config["mic_device"]
        self.output_device = config["output_device"]
        self.monitor_device = config["monitor_device"]
        self.monitor_enabled = config["monitor_enabled"]

        self.monitor_stream = None

        self.init_device_selection()
        self.init_scroll_area()
        self.load_sounds()
        self.add_add_button()
        self.add_stop_button()

        self.active_sounds = []
        self.fs = 44100

        self.start_streams()

    # gui

    def init_device_selection(self):
        self.devices = sd.query_devices()
        mic_names = [d["name"] for d in self.devices if d["max_input_channels"] > 0]
        out_names = [d["name"] for d in self.devices if d["max_output_channels"] > 0]

        # mic
        self.layout.addWidget(QLabel("Microphone Device:"))
        self.mic_combo = QComboBox()
        self.mic_combo.addItems(mic_names)
        if self.mic_device in mic_names:
            self.mic_combo.setCurrentText(self.mic_device)
        self.layout.addWidget(self.mic_combo)

        # out
        self.layout.addWidget(QLabel("Output Device (VB-Cable Input):"))
        self.output_combo = QComboBox()
        self.output_combo.addItems(out_names)
        if self.output_device in out_names:
            self.output_combo.setCurrentText(self.output_device)
        self.layout.addWidget(self.output_combo)

        # check monitor
        self.monitor_checkbox = QCheckBox("Enable Monitoring (Hear Yourself)")
        self.monitor_checkbox.setChecked(self.monitor_enabled)
        self.monitor_checkbox.stateChanged.connect(self.toggle_monitor)
        self.layout.addWidget(self.monitor_checkbox)

        # monitor
        self.layout.addWidget(QLabel("Monitor Output Device:"))
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItems(out_names)
        if self.monitor_device in out_names:
            self.monitor_combo.setCurrentText(self.monitor_device)
        self.layout.addWidget(self.monitor_combo)

        # save event
        self.mic_combo.currentTextChanged.connect(self.save_config)
        self.output_combo.currentTextChanged.connect(self.save_config)
        self.monitor_combo.currentTextChanged.connect(self.save_config)

    def toggle_monitor(self):
        self.monitor_enabled = self.monitor_checkbox.isChecked()
        self.save_config()
        if self.monitor_enabled:
            self.start_monitor_stream()
        else:
            if self.monitor_stream:
                self.monitor_stream.stop()
                self.monitor_stream.close()
                self.monitor_stream = None

    def init_scroll_area(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_frame = QFrame()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setSpacing(8)
        self.scroll_frame.setLayout(self.scroll_layout)
        scroll.setWidget(self.scroll_frame)
        self.layout.addWidget(scroll)

    def load_sounds(self):
        for sound_file in config["sounds"]:
            if os.path.exists(sound_file):
                data, sr = sf.read(sound_file, dtype="float32")
                data = stereoize(data)
                self.sounds[sound_file] = data
                self.add_sound_button(sound_file)

    def add_sound_button(self, sound_file):
        row = QHBoxLayout()
        play_btn = QPushButton(os.path.basename(sound_file))
        play_btn.clicked.connect(lambda _, f=sound_file: self.trigger_sound(f))
        row.addWidget(play_btn)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(100)
        row.addWidget(slider)
        self.vol_sliders[sound_file] = slider

        self.scroll_layout.addLayout(row)

    def add_add_button(self):
        btn = QPushButton("Add New Sound")
        btn.clicked.connect(self.add_sound)
        self.layout.addWidget(btn)

    def add_stop_button(self):
        stop_btn = QPushButton("STOP ALL SOUNDS")
        stop_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        stop_btn.clicked.connect(self.stop_all_sounds)
        self.layout.addWidget(stop_btn)

    def stop_all_sounds(self):
        self.active_sounds.clear()

    def add_sound(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Sound", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            data, sr = sf.read(file_path, dtype="float32")
            data = stereoize(data)
            self.sounds[file_path] = data
            self.add_sound_button(file_path)
            self.save_config()

    def trigger_sound(self, sound_file):
        vol = self.vol_sliders[sound_file].value() / 100
        self.active_sounds.append([self.sounds[sound_file], 0, vol])

    # config save

    def save_config(self):
        config["mic_device"] = self.mic_combo.currentText()
        config["output_device"] = self.output_combo.currentText()
        config["monitor_device"] = self.monitor_combo.currentText()
        config["monitor_enabled"] = self.monitor_enabled
        config["sounds"] = list(self.sounds.keys())
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except:
            pass

    # audio stream

    def start_monitor_stream(self):
        if self.monitor_stream:
            self.monitor_stream.stop()
            self.monitor_stream.close()

        monitor_name = self.monitor_combo.currentText()
        monitor_idx = [i for i, d in enumerate(self.devices) if d["name"] == monitor_name][0]

        self.monitor_stream = sd.OutputStream(
            samplerate=self.fs, blocksize=1024,
            device=monitor_idx, channels=2, dtype="float32"
        )
        self.monitor_stream.start()

    def start_streams(self):
        mic_idx = [i for i, d in enumerate(self.devices) if d["name"] == self.mic_combo.currentText()][0]
        out_idx = [i for i, d in enumerate(self.devices) if d["name"] == self.output_combo.currentText()][0]

        if self.monitor_enabled:
            self.start_monitor_stream()

        def callback(indata, outdata, frames, time, status):
            stereo_mic = stereoize(indata.copy())
            output = stereo_mic.copy()

            new_active = []
            for sound_array, pos, vol in self.active_sounds:
                chunk = sound_array[pos:pos + frames] * vol
                l = min(len(chunk), frames)
                output[:l] += chunk[:l]
                if pos + frames < len(sound_array):
                    new_active.append([sound_array, pos + frames, vol])
            self.active_sounds = new_active

            output = np.clip(output, -1, 1)
            outdata[:] = output

            if self.monitor_enabled and self.monitor_stream:
                self.monitor_stream.write(output.copy())

        self.stream = sd.Stream(
            samplerate=self.fs, blocksize=1024,
            device=(mic_idx, out_idx),
            channels=2, dtype="float32",
            callback=callback
        )
        self.stream.start()


# run the app
app = QApplication([])
window = Soundboard()
window.show()
sys.exit(app.exec())
