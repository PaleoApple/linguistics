import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import pyaudio

# --- Config ---
RATE = 44100
CHUNK = 2048
CHANNELS = 1

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Time vs Frequency")
plot = win.addPlot()

img = pg.ImageItem()
plot.addItem(img)

# Color Map
pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
color = np.array([[0,0,0], [80,0,80], [255,0,0], [255,255,0], [255,255,255]])
lut = pg.ColorMap(pos, color).getLookupTable(0.0, 1.0, 256)
img.setLookupTable(lut)

# --- DIMENSIONS ---
time_steps = 54  # Now X-axis (Horizontal)
freq_bins = 372  # Now Y-axis (Vertical)

# Data shape (X, Y) -> (Time, Freq)
data = np.zeros((time_steps, freq_bins))

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# --- RAW AXIS MAPPING ---
x_axis = plot.getAxis('bottom') # Time
y_axis = plot.getAxis('left')   # Frequency

# Bottom (X) is Time: 0 is the past, 54 is 'Now'
x_axis.setTicks([[(0, '0 s'), (time_steps, '-2.5 s')]])
x_axis.setLabel('Time')

# Left (Y) is Frequency: 0 is 0Hz, 372 is 8kHz
y_axis.setTicks([[(0, '0 Hz'), (freq_bins, '8k Hz')]])
y_axis.setLabel('Frequency')

plot.setXRange(0, time_steps, padding=0)
plot.setYRange(0, freq_bins, padding=0)
plot.setMouseEnabled(x=False, y=False)

def update():
    global data
    try:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        
        # FFT
        magnitude = np.abs(np.fft.rfft(samples * np.hanning(CHUNK)))
        db_slice = 20 * np.log10(magnitude + 1e-10)
        
        # Truncate to 8kHz (372 bins)
        db_slice_truncated = db_slice[:freq_bins]
        
        # Roll on Axis 0 (Time/X) and insert new column at the end (Right side)
        data = np.roll(data, -1, axis=0)
        data[-1, :] = db_slice_truncated
        
        img.setImage(data, autoLevels=False, levels=(50, 140))
        
    except Exception as e:
        print(f"Error: {e}")

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    app.exec()
