#!/usr/bin/env python3
# live_fft_bt_flux_direct.py — Classic BT (SPP) Live FFT with Logging + Playback
# Preconfigured for EC:E3:34:66:8F:72 on RFCOMM channel 1 (ESP32 BluetoothSerial).
#
# Requires: sudo apt install python3-bluez bluez ; pip install --user pyqtgraph numpy
# Run:      python3 live_fft_bt_flux_direct.py
#           (close the Bluetooth Settings window first)

import argparse
import threading
import time
import math
import queue
import socket
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

try:
    import bluetooth  # PyBluez
except Exception as e:
    raise SystemExit("PyBluez missing. Install with: sudo apt install python3-bluez") from e

# -------- Defaults locked to your device --------
DEFAULT_MAC  = "EC:E3:34:66:8F:72"
DEFAULT_PORT = 1

# ------------- Visualizer config ----------------
FALLBACK_HZ      = 400.0
BUFFER_SIZE      = 4096
PLOT_INTERVAL_MS = 20
FFT_UPDATE_EVERY = 3
USE_HIGHPASS     = True
HIGHPASS_ALPHA   = 0.995
REMOVE_MEAN      = True
WINDOW_FUNC      = np.hanning
CONNECT_TIMEOUT  = 4.0
SOCK_TIMEOUT     = 0.03
LOG_DIR          = "logs"
# ------------------------------------------------

def high_pass(signal: np.ndarray, alpha: float = HIGHPASS_ALPHA) -> np.ndarray:
    if len(signal) == 0:
        return signal
    y = np.zeros_like(signal, dtype=float)
    for i in range(1, len(signal)):
        y[i] = alpha * (y[i-1] + signal[i] - signal[i-1])
    return y

class SPPIngest(threading.Thread):
    def __init__(self, out_q: "queue.Queue[tuple]", mac: str, port: int):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.stop_flag = threading.Event()
        self.sock = None
        self.buf = bytearray()
        self.mac = mac
        self.port = port

    def run(self):
        while not self.stop_flag.is_set():
            try:
                print(f"[BT] Connecting to {self.mac}:{self.port} ... (close Bluetooth Settings if open)")
                s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                s.settimeout(CONNECT_TIMEOUT)
                s.connect((self.mac, self.port))
                print("[BT] Connected.")
                s.settimeout(SOCK_TIMEOUT)
                self.sock = s
                self._read_loop()
            except Exception as e:
                print(f"[BT] Connect/read error: {e}")
                time.sleep(0.6)
            finally:
                try:
                    if self.sock:
                        self.sock.close()
                        print("[BT] Disconnected.")
                except Exception:
                    pass
                self.sock = None
                self.buf.clear()

    def _read_loop(self):
        while not self.stop_flag.is_set():
            try:
                chunk = self.sock.recv(1024)
                if not chunk:
                    raise OSError("closed")
                self.buf.extend(chunk)
                now = time.monotonic()
                while b"\n" in self.buf:
                    line, _, rest = self.buf.partition(b"\n")
                    self.buf = bytearray(rest)
                    s = line.decode("utf-8", "ignore").strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) == 4 and parts[0] == "a":
                        ax, ay, az = int(parts[1]), int(parts[2]), int(parts[3])
                        try:
                            self.out_q.put_nowait((now, ax, ay, az))
                        except queue.Full:
                            pass
            except (socket.timeout, TimeoutError):
                continue
            except Exception:
                break

    def stop(self):
        self.stop_flag.set()

class PlaybackIngest(threading.Thread):
    def __init__(self, filepath: Path, out_q: "queue.Queue[tuple]"):
        super().__init__(daemon=True)
        self.filepath = Path(filepath)
        self.out_q = out_q
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.speed = 1.0

    def set_speed(self, speed: float):
        self.speed = max(0.01, float(speed))

    def pause(self, paused: bool):
        if paused: self.pause_flag.set()
        else: self.pause_flag.clear()

    def run(self):
        try:
            with self.filepath.open('r', newline='') as f:
                rdr = csv.reader(f)
                rows = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]))
                        for r in rdr if r and not r[0].startswith('#') and len(r) >= 4]
            if len(rows) < 2: return
            t0 = time.monotonic(); tbase = rows[0][0]
            for t_rel, x, y, z in rows:
                if self.stop_flag.is_set(): break
                while self.pause_flag.is_set() and not self.stop_flag.is_set():
                    time.sleep(0.05)
                target = (t_rel - tbase) / self.speed
                while not self.stop_flag.is_set():
                    elapsed = time.monotonic() - t0
                    dt = target - elapsed
                    if dt <= 0: break
                    time.sleep(min(0.02, dt))
                now = time.monotonic()
                try: self.out_q.put_nowait((now, x, y, z))
                except queue.Full: pass
        except Exception:
            return

    def stop(self):
        self.stop_flag.set()

class SampleRateEMA:
    def __init__(self, init_hz=FALLBACK_HZ, alpha=0.9):
        self.alpha = alpha; self.hz = float(init_hz)
        self._last_sec = int(time.monotonic()); self._count = 0
    def tick(self, n=1):
        t = int(time.monotonic())
        if t != self._last_sec:
            inst = self._count / max(1e-9, (t - self._last_sec))
            self.hz = self.alpha * self.hz + (1 - self.alpha) * inst
            self._last_sec = t; self._count = 0
        self._count += n; return self.hz

class MainUI(QtWidgets.QWidget):
    def __init__(self, mac, port):
        super().__init__()
        self.setWindowTitle("flux_vib_sensor — Live FFT (BT Classic) with Logging + Playback")
        self.resize(1300, 900)
        self.mac, self.port = mac, port

        self.mode = 'live'
        self.q = queue.Queue(maxsize=20000)
        self.ingest = None
        self.log_fp = None
        self.log_writer = None
        self.log_t0 = None

        self.btn_live = QtWidgets.QPushButton("Back to Live")
        self.btn_open = QtWidgets.QPushButton("Open Log…")
        self.btn_play = QtWidgets.QPushButton("▶ Play"); self.btn_play.setCheckable(True)
        self.speed = QtWidgets.QComboBox(); self.speed.addItems(["0.25×","0.5×","1×","2×","4×"]); self.speed.setCurrentText("1×")
        self.btn_log = QtWidgets.QPushButton("● Start Logging"); self.btn_log.setCheckable(True)
        self.lbl_status = QtWidgets.QLabel("Status: idle")

        top = QtWidgets.QHBoxLayout()
        for w in (self.btn_live, self.btn_open, self.btn_play, QtWidgets.QLabel("Speed:"), self.speed):
            top.addWidget(w)
        top.addStretch(1); top.addWidget(self.btn_log); top.addSpacing(10); top.addWidget(self.lbl_status)

        pg.setConfigOptions(antialias=True, background='k', foreground='w')
        self.win = pg.GraphicsLayoutWidget()
        self.time_curves, self.fft_curves = [], []
        self.buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(3)]
        for i, axis in enumerate(['X','Y','Z']):
            p = self.win.addPlot(row=i, col=0, title=f"Time: Axis {axis}")
            p.showGrid(x=True, y=True); self.time_curves.append(p.plot())
        for i, axis in enumerate(['X','Y','Z']):
            p = self.win.addPlot(row=3+i, col=0, title=f"FFT: Axis {axis}")
            p.showGrid(x=True, y=True); self.fft_curves.append(p.plot())

        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.win)

        self.btn_log.toggled.connect(self.on_toggle_log)
        self.btn_open.clicked.connect(self.on_open_log)
        self.btn_play.toggled.connect(self.on_toggle_play)
        self.btn_live.clicked.connect(self.on_back_to_live)
        self.speed.currentTextChanged.connect(self.on_speed_change)

        self.sps = SampleRateEMA(init_hz=FALLBACK_HZ); self.plot_counter = 0
        self.start_live()

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.on_timer); self.timer.start(PLOT_INTERVAL_MS)

    def start_live(self):
        self.stop_ingest()
        self.mode = 'live'
        self.lbl_status.setText(f"Status: connecting to {self.mac}:{self.port} …")
        self.ingest = SPPIngest(self.q, self.mac, self.port)
        self.ingest.start()
        self.btn_play.setEnabled(False); self.btn_open.setEnabled(True); self.btn_log.setEnabled(True)

    def start_playback(self, filepath: Path):
        self.stop_ingest(); self.mode = 'playback'
        self.lbl_status.setText(f"Status: playback {Path(filepath).name}")
        pb = PlaybackIngest(filepath, self.q); pb.set_speed(self._speed_value())
        self.ingest = pb; self.ingest.start()
        self.btn_play.setEnabled(True); self.btn_open.setEnabled(True); self.btn_log.setEnabled(False)

    def stop_ingest(self):
        if self.ingest:
            try: self.ingest.stop()
            except Exception: pass
            self.ingest = None
        try:
            while True: self.q.get_nowait()
        except queue.Empty:
            pass

    def on_toggle_log(self, checked: bool):
        if checked:
            Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
            default = Path(LOG_DIR) / f"flux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save log as…", str(default), "CSV (*.csv)")
            if not path: self.btn_log.setChecked(False); return
            try:
                self.log_fp = open(path, 'w', newline=''); self.log_writer = csv.writer(self.log_fp)
                self.log_fp.write("# flux_vib_sensor log (BT Classic / SPP)\n# t,ax,ay,az\n")
                self.log_t0 = None; self.btn_log.setText("■ Stop Logging")
                self.lbl_status.setText(f"Status: logging → {Path(path).name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open file:\n{e}")
                self.btn_log.setChecked(False)
        else:
            self._close_log()

    def _close_log(self):
        if self.log_fp:
            try: self.log_fp.flush(); self.log_fp.close()
            except Exception: pass
        self.log_fp = None; self.log_writer = None; self.log_t0 = None
        self.btn_log.setText("● Start Logging")
        if self.mode == 'live': self.lbl_status.setText("Status: live (connected)")

    def on_open_log(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open log…", LOG_DIR, "CSV (*.csv)")
        if path: self.start_playback(Path(path))

    def on_toggle_play(self, checked: bool):
        if self.mode != 'playback' or not isinstance(self.ingest, PlaybackIngest):
            self.btn_play.setChecked(False); return
        if checked: self.ingest.pause(False); self.btn_play.setText("⏸ Pause")
        else: self.ingest.pause(True); self.btn_play.setText("▶ Play")

    def on_speed_change(self, _):
        sp = self._speed_value()
        if self.mode == 'playback' and isinstance(self.ingest, PlaybackIngest):
            self.ingest.set_speed(sp)

    def _speed_value(self) -> float:
        txt = self.speed.currentText().replace("×","")
        try: return float(txt)
        except Exception: return 1.0

    def on_back_to_live(self):
        self.start_live()

    def on_timer(self):
        drained = 0
        try:
            while True:
                ts, ax, ay, az = self.q.get_nowait()
                if self.log_writer is not None:
                    if self.log_t0 is None: self.log_t0 = ts
                    t_rel = ts - self.log_t0
                    try: self.log_writer.writerow([f"{t_rel:.6f}", ax, ay, az])
                    except Exception: pass
                self.buffers[0].append(ax); self.buffers[1].append(ay); self.buffers[2].append(az)
                drained += 1
        except queue.Empty:
            pass

        if self.mode == 'live' and drained:
            self.lbl_status.setText("Status: live (connected)")

        if drained:
            self.sps.tick(drained); self.plot_counter += 1

        if any(len(b) for b in self.buffers):
            for i in range(3):
                self.time_curves[i].setData(np.fromiter(self.buffers[i], dtype=float))

        if self.plot_counter >= FFT_UPDATE_EVERY and all(len(b) == BUFFER_SIZE for b in self.buffers):
            fs = max(1.0, self.sps.hz)
            freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/fs)
            for i in range(3):
                raw = np.fromiter(self.buffers[i], dtype=float)
                sig = high_pass(raw) if USE_HIGHPASS else raw
                if REMOVE_MEAN: sig = sig - np.mean(sig)
                window = WINDOW_FUNC(len(sig))
                fft_mag = np.abs(np.fft.rfft(sig * window))
                denom = np.max(fft_mag) if np.max(fft_mag) > 0 else 1.0
                self.fft_curves[i].setData(freqs, fft_mag/denom)
            self.plot_counter = 0

    def closeEvent(self, e):
        try: self.stop_ingest(); self._close_log()
        finally: super().closeEvent(e)

def parse_args():
    ap = argparse.ArgumentParser(description="Classic BT SPP Live FFT (preconfigured)")
    ap.add_argument("--mac", default=DEFAULT_MAC, help="Target BR/EDR MAC (default is preconfigured).")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="RFCOMM channel (default 1).")
    return ap.parse_args()

def main():
    args = parse_args()
    app = QtWidgets.QApplication([])
    ui = MainUI(mac=args.mac, port=args.port)
    ui.show()
    QtWidgets.QApplication.instance().exec()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
