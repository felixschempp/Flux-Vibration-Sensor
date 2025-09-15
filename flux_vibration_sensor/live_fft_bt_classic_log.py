#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux Vibration GUI: live plot + FFT, CSV logging, playback with seek/speed.
Author: you + ChatGPT
"""

import os, re, csv, time, sys
import numpy as np
from collections import deque

# ---- Qt/PG ----
try:
    from pyqtgraph.Qt import QtWidgets, QtCore
except ImportError:
    from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ---- Serial ----
import serial
from serial.tools import list_ports

# ====== CONFIG ======
DEFAULT_PORT   = '/dev/rfcomm0'
DEFAULT_BAUD   = 115200
BUFFER_SIZE    = 1024
PLOT_UPDATE_MS = 30    # plot refresh timer
READ_INTERVAL  = 10    # ms timer for serial/playback feeding
DEFAULT_FS     = 400.0 # Hz (used for plotting & header)
USE_HIGHPASS   = True
HIGHPASS_ALPHA = 0.995 # ~1 Hz cutoff at fs≈400 Hz
REMOVE_MEAN    = True
EPSILON        = 1e-6

LINE_REGEX = re.compile(r'a\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)')

# ====== DSP ======
def high_pass(signal, alpha=HIGHPASS_ALPHA):
    y = np.zeros_like(signal, dtype=float)
    for i in range(1, len(signal)):
        y[i] = alpha * (y[i-1] + signal[i] - signal[i-1])
    return y

# ====== App ======
class VibApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux Vibration – Live/Log/Playback")
        self.resize(1200, 900)

        # --- State ---
        self.mode = 'idle'        # 'idle' | 'live' | 'playback'
        self.fs = DEFAULT_FS
        self.ser = None
        self.log_fp = None
        self.log_writer = None
        self.log_t0 = None
        self.play_data = None     # dict with 't','ax','ay','az'
        self.play_idx = 0
        self.play_speed = 1.0

        # Buffers for plotting
        self.buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(3)]  # ax, ay, az

        # --- UI ---
        self._build_ui()

        # --- Timers ---
        self.read_timer = QtCore.QTimer(self)     # feeds data (serial or playback)
        self.read_timer.timeout.connect(self.feed_loop)
        self.read_timer.start(READ_INTERVAL)

        self.plot_timer = QtCore.QTimer(self)     # refresh plots
        self.plot_timer.timeout.connect(self.refresh_plots)
        self.plot_timer.start(PLOT_UPDATE_MS)

    # ---------- UI ----------
    def _build_ui(self):
        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)
        v.setContentsMargins(8,8,8,8)
        v.setSpacing(6)

        # Toolbar / Controls
        ctl = QtWidgets.QHBoxLayout()
        ctl.setSpacing(8)
        v.addLayout(ctl)

        # Port
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        self._refresh_ports()
        self.port_combo.setEditText(DEFAULT_PORT)
        self.baud_spin = QtWidgets.QSpinBox()
        self.baud_spin.setRange(1200, 2000000)
        self.baud_spin.setValue(DEFAULT_BAUD)
        ctl.addWidget(QtWidgets.QLabel("Port:"))
        ctl.addWidget(self.port_combo, 2)
        ctl.addWidget(QtWidgets.QLabel("Baud:"))
        ctl.addWidget(self.baud_spin, 1)

        # Connect / Disconnect
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_disconnect.clicked.connect(self.on_disconnect)
        ctl.addWidget(self.btn_connect)
        ctl.addWidget(self.btn_disconnect)

        # Logging
        self.btn_start_log = QtWidgets.QPushButton("Start Log")
        self.btn_stop_log  = QtWidgets.QPushButton("Stop Log")
        self.btn_stop_log.setEnabled(False)
        self.btn_start_log.clicked.connect(self.on_start_log)
        self.btn_stop_log.clicked.connect(self.on_stop_log)
        ctl.addSpacing(12)
        ctl.addWidget(self.btn_start_log)
        ctl.addWidget(self.btn_stop_log)

        # Playback
        self.btn_load_log = QtWidgets.QPushButton("Load Log…")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_load_log.clicked.connect(self.on_load_log)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        ctl.addSpacing(12)
        ctl.addWidget(self.btn_load_log)
        ctl.addWidget(self.btn_play)
        ctl.addWidget(self.btn_pause)

        # Seek + speed
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(0)
        self.seek_slider.valueChanged.connect(self.on_seek)
        self.seek_slider.setEnabled(False)
        ctl.addWidget(QtWidgets.QLabel("Seek:"))
        ctl.addWidget(self.seek_slider, 4)

        self.speed_combo = QtWidgets.QComboBox()
        for s in ["0.25×", "0.5×", "1×", "1.5×", "2×", "4×"]:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.currentTextChanged.connect(self.on_speed)
        ctl.addWidget(QtWidgets.QLabel("Speed:"))
        ctl.addWidget(self.speed_combo)

        # Sample rate (for plotting + header)
        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(1.0, 50000.0)
        self.fs_spin.setDecimals(1)
        self.fs_spin.setValue(self.fs)
        self.fs_spin.valueChanged.connect(self.on_fs_changed)
        ctl.addWidget(QtWidgets.QLabel("fs [Hz]:"))
        ctl.addWidget(self.fs_spin)

        # Status
        self.status = QtWidgets.QLabel("Ready.")
        ctl.addWidget(self.status, 3)

        # Plots
        self.pg = pg.GraphicsLayoutWidget()
        v.addWidget(self.pg, 1)

        # Time plot
        self.pg.addLabel("Time-Domain Acceleration", col=0)
        self.time_plot = self.pg.addPlot(row=1, col=0, title="Acceleration over Time")
        self.time_plot.addLegend()
        self.time_curves = [
            self.time_plot.plot(pen='r', name='ax'),
            self.time_plot.plot(pen='g', name='ay'),
            self.time_plot.plot(pen='b', name='az')
        ]

        # FFT plots
        self.pg.addLabel("Frequency-Domain (FFT)", row=2, col=0)
        self.fft_curves = []
        self.fft_plots  = []
        for i, axis in enumerate('XYZ'):
            p = self.pg.addPlot(row=3+i, col=0, title=f"FFT: Axis {axis}")
            c = p.plot(pen='w')
            self.fft_plots.append(p)
            self.fft_curves.append(c)

        self._rebuild_freq_axis()

    def _refresh_ports(self):
        self.port_combo.clear()
        # Show obvious serial devices, but keep it editable
        ports = [p.device for p in list_ports.comports()]
        # Add rfcomm as likely candidate
        for cand in ['/dev/rfcomm0', '/dev/ttyBLE0']:
            if cand not in ports and os.path.exists(cand):
                ports.append(cand)
        self.port_combo.addItems(ports or [DEFAULT_PORT])

    def _rebuild_freq_axis(self):
        self.freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/self.fs)

    # ---------- Actions ----------
    def on_fs_changed(self, val):
        self.fs = float(val)
        self._rebuild_freq_axis()

    def on_speed(self, txt):
        self.play_speed = float(txt.replace("×", ""))

    def on_connect(self):
        if self.ser:
            return
        port = self.port_combo.currentText().strip()
        baud = int(self.baud_spin.value())
        try:
            self.ser = serial.Serial(port, baud, timeout=0)
            self.mode = 'live'
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.status.setText(f"Connected: {port} @ {baud}")
        except Exception as e:
            self.status.setText(f"Serial error: {e}")
            self.ser = None
            self.mode = 'idle'

    def on_disconnect(self):
        self._stop_logging()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.mode = 'idle'
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.status.setText("Disconnected.")

    def on_start_log(self):
        if self.mode != 'live':
            self.status.setText("Start Log: connect in Live mode first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save log CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.log_fp = open(path, 'w', newline='')
            self.log_writer = csv.writer(self.log_fp)
            # header
            self.log_fp.write(f"# fs={self.fs}\n")
            self.log_writer.writerow(['t','ax','ay','az'])
            self.log_t0 = time.perf_counter()
            self.btn_start_log.setEnabled(False)
            self.btn_stop_log.setEnabled(True)
            self.status.setText(f"Logging → {os.path.basename(path)}")
        except Exception as e:
            self.status.setText(f"Log open failed: {e}")
            self.log_fp = None
            self.log_writer = None

    def on_stop_log(self):
        self._stop_logging()

    def _stop_logging(self):
        if self.log_fp:
            try:
                self.log_fp.flush()
                self.log_fp.close()
            except Exception:
                pass
        self.log_fp = None
        self.log_writer = None
        self.log_t0 = None
        self.btn_start_log.setEnabled(True)
        self.btn_stop_log.setEnabled(False)
        if self.mode == 'live':
            self.status.setText("Logging stopped.")

    def on_load_log(self):
        self.on_disconnect()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open log CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            fs = self._read_header_fs(path) or DEFAULT_FS
            data = self._read_csv_data(path)
            if data['t'].size == 0:
                raise RuntimeError("Empty log.")
            self.fs = float(fs)
            self.fs_spin.setValue(self.fs)
            self.play_data = data
            self.play_idx = 0
            self.mode = 'playback'
            self.seek_slider.setEnabled(True)
            self.seek_slider.setMaximum(len(data['t'])-1)
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.status.setText(f"Loaded {os.path.basename(path)} (N={len(data['t'])}, fs≈{self.fs} Hz)")
            # clear plots/buffers
            for b in self.buffers:
                b.clear()
        except Exception as e:
            self.status.setText(f"Load failed: {e}")
            self.mode = 'idle'
            self.play_data = None
            self.seek_slider.setEnabled(False)

    def _read_header_fs(self, path):
        fs = None
        with open(path, 'r') as f:
            for _ in range(5):  # first few lines
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith("#"):
                    if "fs=" in line:
                        try:
                            fs = float(line.strip().split("fs=")[1])
                        except Exception:
                            pass
                else:
                    # rewind to start of data
                    f.seek(pos)
                    break
        return fs

    def _read_csv_data(self, path):
        # read columns t,ax,ay,az (skip header lines starting with #)
        t, ax, ay, az = [], [], [], []
        with open(path, 'r') as f:
            rdr = csv.reader(row for row in f if not row.startswith('#'))
            header = next(rdr, None)
            # accept flexible header order but expect 4 columns
            for row in rdr:
                if len(row) < 4:
                    continue
                try:
                    ti, xi, yi, zi = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                except Exception:
                    continue
                t.append(ti); ax.append(xi); ay.append(yi); az.append(zi)
        return {'t': np.asarray(t), 'ax': np.asarray(ax), 'ay': np.asarray(ay), 'az': np.asarray(az)}

    def on_play(self):
        if self.mode != 'playback' or self.play_data is None:
            return
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.status.setText("Playback: playing…")

    def on_pause(self):
        if self.mode != 'playback':
            return
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.status.setText("Playback: paused.")

    def on_seek(self, idx):
        if self.mode != 'playback' or self.play_data is None:
            return
        self.play_idx = int(idx)
        # Also update buffers immediately for visual feedback
        self._prime_buffers_from_playback()

    # ---------- Data feeding ----------
    def feed_loop(self):
        if self.mode == 'live':
            self._feed_from_serial()
        elif self.mode == 'playback':
            self._feed_from_playback()

    def _feed_from_serial(self):
        if not self.ser:
            return
        try:
            # read whatever is there; don't block
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                m = LINE_REGEX.match(line)
                if not m:
                    continue
                vals = list(map(int, m.groups()))  # ax, ay, az (raw)
                for i in range(3):
                    self.buffers[i].append(vals[i])
                # log raw ints with relative timestamp
                if self.log_writer and self.log_t0 is not None:
                    t = time.perf_counter() - self.log_t0
                    self.log_writer.writerow([f"{t:.6f}", vals[0], vals[1], vals[2]])
        except Exception as e:
            self.status.setText(f"Serial read error: {e}")
            # optional: auto-disconnect
            # self.on_disconnect()

    def _prime_buffers_from_playback(self):
        """Fill buffers around current play_idx for smooth plots."""
        if self.play_data is None: return
        start = max(0, self.play_idx - BUFFER_SIZE)
        end   = min(len(self.play_data['t']), self.play_idx)
        segX = self.play_data['ax'][start:end]
        segY = self.play_data['ay'][start:end]
        segZ = self.play_data['az'][start:end]
        for b in self.buffers: b.clear()
        for xi, yi, zi in zip(segX, segY, segZ):
            self.buffers[0].append(xi)
            self.buffers[1].append(yi)
            self.buffers[2].append(zi)

    def _feed_from_playback(self):
        if self.play_data is None:
            return
        # Only advance when "playing"
        if not self.btn_pause.isEnabled():
            return

        # Estimate how many samples to feed each tick from desired speed and fs
        # samples per ms = fs/1000; over READ_INTERVAL ms → chunk
        spms = self.fs / 1000.0
        chunk = max(1, int(round(spms * READ_INTERVAL * self.play_speed)))

        end = min(len(self.play_data['t']), self.play_idx + chunk)
        for idx in range(self.play_idx, end):
            self.buffers[0].append(float(self.play_data['ax'][idx]))
            self.buffers[1].append(float(self.play_data['ay'][idx]))
            self.buffers[2].append(float(self.play_data['az'][idx]))
        self.play_idx = end
        if self.seek_slider.isEnabled():
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.play_idx)
            self.seek_slider.blockSignals(False)
        if self.play_idx >= len(self.play_data['t']):
            # reached end
            self.btn_pause.setEnabled(False)
            self.btn_play.setEnabled(True)
            self.status.setText("Playback: finished.")

    # ---------- Plotting ----------
    def refresh_plots(self):
        # Need full buffers for FFT; but update time plot anyhow
        have_full = all(len(b) == BUFFER_SIZE for b in self.buffers)

        for i in range(3):
            arr = np.array(self.buffers[i], dtype=float)
            if arr.size == 0:
                continue

            # Time-domain processing (view only)
            sig = arr.copy()
            if REMOVE_MEAN and sig.size > 0:
                sig -= np.mean(sig)
            if USE_HIGHPASS and sig.size > 1:
                sig = high_pass(sig)

            # Time plot (x in seconds for context)
            t_axis = np.arange(sig.size) / self.fs
            self.time_curves[i].setData(t_axis, sig)

            # FFT
            if have_full:
                windowed = sig * np.hanning(sig.size)
                fft_mag = np.abs(np.fft.rfft(windowed))
                fft_mag /= (np.max(fft_mag) + EPSILON)
                self.fft_curves[i].setData(self.freqs, fft_mag)
            else:
                # clear until full
                self.fft_curves[i].setData([], [])

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = VibApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()