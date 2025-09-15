# Flux-Vibration-Sensor

<img width="1961" height="1060" alt="flux_vibration_module_v1_2025-Aug-27_09-00-11AM-000_CustomizedView2098348405" src="https://github.com/user-attachments/assets/dcd080d4-b787-46e3-866c-7ae9837662eb" />

# Flux Vibration Sensor – Live FFT GUI

This repo provides Python tools to **stream, visualize, log, and replay accelerometer data** (e.g. from an ESP32 over Bluetooth Classic SPP).  
It uses **PyQtGraph** for fast plotting, with both **time-domain** and **frequency-domain (FFT)** views.  

---

## Features

- 📡 Live Bluetooth Classic SPP ingestion (ESP32 `BluetoothSerial`)  
- 📈 Real-time plotting of 3-axis accelerometer data  
- 🔊 FFT spectrum analysis with optional high-pass filter and DC removal  
- 💾 CSV logging of captured data  
- 🎞 Playback mode with adjustable speed & seek support  
- 🔌 Multiple backends:  
  - `live_fft_bt.py` → simple real-time plotter (serial via `/dev/rfcomm0`)  
  - `live_fft_bt_classic_log.py` → serial + logging + playback  
  - `live_fft_bt_classic_log_v2.py` → improved, direct Bluetooth (PyBluez) + logging + playback  

---

## Requirements

System packages:

```bash
sudo apt install python3-pyqt5 python3-pyqtgraph python3-serial python3-bluez bluez
