# import serial
# import numpy as np
# from collections import deque
# import pyqtgraph as pg
# import re

# # ---- Qt Compatibility ----
# try:
#     from pyqtgraph.Qt import QtWidgets, QtCore
# except ImportError:
#     from PyQt5 import QtWidgets, QtCore
# # --------------------------

# # ----- CONFIG -----
# PORT = '/dev/rfcomm0'
# BAUD = 115200
# BUFFER_SIZE = 1024
# PLOT_UPDATE_EVERY = 10
# fs = 400  # sampling rate in Hz
# USE_HIGHPASS = True
# HIGHPASS_ALPHA = 0.995  # ~1 Hz cutoff at 400 Hz
# # -------------------

# # ----- High-pass filter -----
# def high_pass(signal, alpha=HIGHPASS_ALPHA):
#     y = np.zeros_like(signal, dtype=float)
#     for i in range(1, len(signal)):
#         y[i] = alpha * (y[i-1] + signal[i] - signal[i-1])
#     return y

# # ----- Setup serial and buffers -----
# ser = serial.Serial(PORT, BAUD, timeout=1)
# buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(3)]  # ax, ay, az

# # ----- Qt window setup -----
# app = QtWidgets.QApplication([])
# win = pg.GraphicsLayoutWidget(show=True, title="MPU6050 Real-Time Data + FFT")
# win.resize(1000, 800)

# # Time-domain plot
# win.addLabel("Time-Domain Acceleration", col=0)
# time_plot = win.addPlot(row=1, col=0, title="Acceleration over Time")
# time_plot.addLegend()
# time_curves = [
#     time_plot.plot(pen='r', name='ax'),
#     time_plot.plot(pen='g', name='ay'),
#     time_plot.plot(pen='b', name='az')
# ]

# # Frequency-domain plots
# win.addLabel("Frequency-Domain (FFT)", row=2, col=0)
# fft_plots = []
# fft_curves = []
# freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/fs)

# for i, axis in enumerate(['X', 'Y', 'Z']):
#     p = win.addPlot(row=3+i, col=0, title=f"FFT: Axis {axis}")
#     c = p.plot()
#     fft_plots.append(p)
#     fft_curves.append(c)

# plot_counter = 0

# # ----- Main update loop -----
# def update():
#     global plot_counter

#     while ser.in_waiting:
#         line = ser.readline().decode('utf-8', errors='ignore').strip()
#         match = re.match(r'a\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)', line)
#         if match:
#             values = list(map(int, match.groups()))
#             for i in range(3):
#                 buffers[i].append(values[i])
#             plot_counter += 1

#     if plot_counter >= PLOT_UPDATE_EVERY and all(len(b) == BUFFER_SIZE for b in buffers):
#         # Time domain plot update
#         for i in range(3):
#             time_curves[i].setData(list(buffers[i]))

#         # Frequency domain update
#         for i in range(3):
#             raw_signal = np.array(buffers[i])
#             signal = high_pass(raw_signal) if USE_HIGHPASS else raw_signal
#             windowed = signal * np.hanning(len(signal))
#             fft_mag = np.abs(np.fft.rfft(windowed))
#             fft_mag /= np.max(fft_mag)
#             fft_curves[i].setData(freqs, fft_mag)

#         plot_counter = 0

# # ----- Timer -----
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(10)

# # ----- Run Qt event loop -----
# QtWidgets.QApplication.instance().exec()

#///////////////////////////////////////////////////////////////

#/////////////////////////////////////////////////////////////////////////


# # --- live_fft_bt_safe.py (drop-in replacement) ---
# import os, time, errno, subprocess
# import serial
# import numpy as np
# from collections import deque
# import pyqtgraph as pg
# import re

# # ---- Qt Compatibility ----
# try:
#     from pyqtgraph.Qt import QtWidgets, QtCore
# except ImportError:
#     from PyQt5 import QtWidgets, QtCore

# # ===== CONFIG =====
# # Your ESP32 Classic BT details
# BT_MAC        = "EC:E3:34:66:8F:72"   # <- your device MAC
# RFCOMM_IDX    = 0                     # -> /dev/rfcomm0
# SPP_CHANNEL   = "1"                   # most ESP32 SPP examples use channel 1

# # Serial/plot parameters
# BAUD = 115200
# PORT = f"/dev/rfcomm{RFCOMM_IDX}"
# BUFFER_SIZE = 1024
# PLOT_UPDATE_EVERY = 10
# fs = 400  # Sampling frequency in Hz

# USE_HIGHPASS = True
# HIGHPASS_ALPHA = 0.995  # ~1 Hz cutoff
# REMOVE_MEAN = True
# EPSILON = 1e-6

# LINE_REGEX = re.compile(r'a\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)')

# # ===== Helpers: RFCOMM + Serial reopen =====
# def ensure_rfcomm(idx=RFCOMM_IDX, mac=BT_MAC, ch=SPP_CHANNEL):
#     """Bind /dev/rfcomm{idx} to a given MAC/channel if not present."""
#     dev = f"/dev/rfcomm{idx}"
#     if not os.path.exists(dev):
#         # Release (ignore errors), then bind
#         subprocess.run(["rfcomm", "release", str(idx)], check=False)
#         subprocess.run(["rfcomm", "bind", str(idx), mac, ch], check=True)
#         # Wait briefly for udev to create the node
#         for _ in range(20):
#             if os.path.exists(dev):
#                 break
#             time.sleep(0.05)
#     return dev

# def reopen_serial(old=None, baud=BAUD):
#     """Close old handle, re-bind rfcomm, reopen serial safely."""
#     try:
#         if old:
#             old.close()
#     except Exception:
#         pass
#     port = ensure_rfcomm()
#     # Short timeouts to keep GUI responsive; no .in_waiting anywhere
#     return serial.Serial(port, baud, timeout=0.1, inter_byte_timeout=0.05)

# # ===== Signal processing =====
# def high_pass(signal, alpha=HIGHPASS_ALPHA):
#     y = np.zeros_like(signal, dtype=float)
#     for i in range(1, len(signal)):
#         y[i] = alpha * (y[i-1] + signal[i] - signal[i-1])
#     return y

# # ===== Setup serial and buffers =====
# ser = None
# try:
#     # Try existing port first; if it isn't there, bind it
#     if not os.path.exists(PORT):
#         ensure_rfcomm()
#     ser = serial.Serial(PORT, BAUD, timeout=0.1, inter_byte_timeout=0.05)
# except Exception:
#     # Fallback: (re)bind and open
#     ser = reopen_serial(None, BAUD)

# recv_buf = bytearray()  # accumulate partial lines across reads
# buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(3)]  # ax, ay, az

# # ===== Qt window setup =====
# app = QtWidgets.QApplication([])
# win = pg.GraphicsLayoutWidget(show=True, title="MPU6050 Real-Time Data + FFT")
# win.resize(1000, 800)

# # Time-domain plot
# win.addLabel("Time-Domain Acceleration", col=0)
# time_plot = win.addPlot(row=1, col=0, title="Acceleration over Time")
# time_plot.addLegend()
# time_curves = [
#     time_plot.plot(pen='r', name='ax'),
#     time_plot.plot(pen='g', name='ay'),
#     time_plot.plot(pen='b', name='az')
# ]

# # Frequency-domain plots
# win.addLabel("Frequency-Domain (FFT)", row=2, col=0)
# fft_curves = []
# freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/fs)
# for i, axis in enumerate(['X', 'Y', 'Z']):
#     p = win.addPlot(row=3+i, col=0, title=f"FFT: Axis {axis}")
#     c = p.plot(pen='w')
#     fft_curves.append(c)

# plot_counter = 0

# # ===== Main update loop (no ser.in_waiting, auto-reconnect) =====
# def update():
#     global plot_counter, recv_buf, ser

#     # Read a few chunks per GUI tick to stay responsive
#     for _ in range(8):
#         try:
#             chunk = ser.read(4096)  # rely on timeout; avoids ioctl(TIOCINQ)
#         except OSError as e:
#             # Handle kernel I/O errors that happen when BT link drops
#             if getattr(e, "errno", None) in (errno.EIO, errno.EHOSTUNREACH, errno.ENETUNREACH):
#                 # Rebind rfcomm and reopen serial; skip rest of this frame
#                 ser = reopen_serial(ser, BAUD)
#                 return
#             raise  # unexpected error → surface it

#         if not chunk:
#             break  # nothing available this moment

#         recv_buf.extend(chunk)

#         # Process any complete lines; keep last partial in recv_buf
#         if b'\n' in recv_buf:
#             lines = recv_buf.split(b'\n')
#             recv_buf = bytearray(lines.pop())  # leftover partial
#             for raw in lines:
#                 line = raw.decode('utf-8', errors='ignore').strip()
#                 m = LINE_REGEX.match(line)
#                 if m:
#                     values = list(map(int, m.groups()))
#                     for i in range(3):
#                         buffers[i].append(values[i])
#                     plot_counter += 1

#     # Update plots when enough new samples gathered and buffers are full
#     if plot_counter >= PLOT_UPDATE_EVERY and all(len(b) == BUFFER_SIZE for b in buffers):
#         for i in range(3):
#             raw_signal = np.array(buffers[i], dtype=float)

#             if REMOVE_MEAN:
#                 raw_signal -= np.mean(raw_signal)

#             signal = high_pass(raw_signal) if USE_HIGHPASS else raw_signal

#             # Time-domain
#             time_curves[i].setData(signal)

#             # Frequency-domain
#             windowed = signal * np.hanning(len(signal))
#             fft_mag = np.abs(np.fft.rfft(windowed))
#             fft_mag /= (np.max(fft_mag) + EPSILON)
#             fft_curves[i].setData(freqs, fft_mag)

#         plot_counter = 0

# # ===== Timer =====
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(10)

# # ===== Run Qt event loop =====
# QtWidgets.QApplication.instance().exec()
#//////////////////////////////////////////////////////////////////////

import serial
import numpy as np
from collections import deque
import pyqtgraph as pg
import re

# ---- Qt Compatibility ----
try:
    from pyqtgraph.Qt import QtWidgets, QtCore
except ImportError:
    from PyQt5 import QtWidgets, QtCore

# ----- CONFIG -----
PORT = '/dev/rfcomm0'
BAUD = 115200
BUFFER_SIZE = 1024
PLOT_UPDATE_EVERY = 10
fs = 400  # Sampling frequency in Hz

USE_HIGHPASS = True
HIGHPASS_ALPHA = 0.995  # ~1 Hz cutoff
REMOVE_MEAN = True      # Remove DC offset before plotting
EPSILON = 1e-6          # For safe FFT normalization

# ----- High-pass filter -----
def high_pass(signal, alpha=HIGHPASS_ALPHA):
    y = np.zeros_like(signal, dtype=float)
    for i in range(1, len(signal)):
        y[i] = alpha * (y[i-1] + signal[i] - signal[i-1])
    return y

# ----- Setup serial and buffers -----
ser = serial.Serial(PORT, BAUD, timeout=1)
buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(3)]  # ax, ay, az

# ----- Qt window setup -----
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="MPU6050 Real-Time Data + FFT")
win.resize(1000, 800)

# Time-domain plot
win.addLabel("Time-Domain Acceleration", col=0)
time_plot = win.addPlot(row=1, col=0, title="Acceleration over Time")
time_plot.addLegend()
time_curves = [
    time_plot.plot(pen='r', name='ax'),
    time_plot.plot(pen='g', name='ay'),
    time_plot.plot(pen='b', name='az')
]

# Frequency-domain plots
win.addLabel("Frequency-Domain (FFT)", row=2, col=0)
fft_curves = []
freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1.0/fs)

for i, axis in enumerate(['X', 'Y', 'Z']):
    p = win.addPlot(row=3+i, col=0, title=f"FFT: Axis {axis}")
    c = p.plot(pen='w')
    fft_curves.append(c)

plot_counter = 0

# ----- Main update loop -----
def update():
    global plot_counter

    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        match = re.match(r'a\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)', line)
        if match:
            values = list(map(int, match.groups()))
            for i in range(3):
                buffers[i].append(values[i])
            plot_counter += 1

    if plot_counter >= PLOT_UPDATE_EVERY and all(len(b) == BUFFER_SIZE for b in buffers):
        for i in range(3):
            raw_signal = np.array(buffers[i], dtype=float)

            # Optional: remove DC offset
            if REMOVE_MEAN:
                raw_signal -= np.mean(raw_signal)

            # Optional: high-pass filtering
            signal = high_pass(raw_signal) if USE_HIGHPASS else raw_signal

            # Update time plot
            time_curves[i].setData(signal)

            # FFT with windowing
            windowed = signal * np.hanning(len(signal))
            fft_mag = np.abs(np.fft.rfft(windowed))
            fft_mag /= (np.max(fft_mag) + EPSILON)  # Normalize safely
            fft_curves[i].setData(freqs, fft_mag)

        plot_counter = 0

# ----- Timer -----
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)

# ----- Run Qt event loop -----
QtWidgets.QApplication.instance().exec()

