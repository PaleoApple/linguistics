import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
RATE = 44100
CHUNK = 2048  
FORMAT = pyaudio.paInt16
CHANNELS = 1
SECONDS_TO_SHOW = 2
COLS = (RATE * SECONDS_TO_SHOW) // CHUNK 

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

num_freq_bins = CHUNK // 2 + 1
spec_data = np.full((num_freq_bins, COLS), -80.0) 

img = ax.imshow(
    spec_data, 
    aspect='auto', 
    origin='lower', 
    extent=[-SECONDS_TO_SHOW, 0, 0, RATE // 2],
    cmap='inferno', 
    vmin=-80, 
    vmax=0,
    animated=True # Required for blitting
)


ax.set_ylim(0, 8000)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Frequency (Hz)")

# Prepare for Blitting
fig.canvas.draw()
background = fig.canvas.copy_from_bbox(ax.bbox)
window = np.hanning(CHUNK)

print("--- Recording Live (Blitting Enabled) ---")

try:
    while True:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # FFT and dB conversion
        magnitude = np.abs(np.fft.rfft(samples * window))
        db_slice = 20 * np.log10(magnitude + 1e-10)

        # Update data array
        spec_data[:, :-1] = spec_data[:, 1:]
        spec_data[:, -1] = db_slice

        # --- Fast Update Steps ---
        img.set_data(spec_data)
        
        # Restore the clean background (no old image)
        fig.canvas.restore_region(background)
        # Redraw just the image
        ax.draw_artist(img)
        # Push it to the screen
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()
        
except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
