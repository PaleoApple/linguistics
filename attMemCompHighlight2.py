import numpy as np
import pyaudio
import threading
import time
import queue
import torch
import re
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- Config ---
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3 
MAX_SAMPLES = RATE * RECORD_SECONDS
DEVICE_INDEX = 7

# ANSI Colors for terminal highlighting
YELLOW = "\033[93m"
BOLD = "\033[1m"
ENDC = "\033[0m"

print("Loading Models... (This may take a moment)")
whisper_model = WhisperModel("tiny", device="cuda", compute_type="int8")

# Official Meta Phonetic Model
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
phonetic_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to("cpu")

data_queue = queue.Queue()
audio_history = np.zeros(MAX_SAMPLES, dtype=np.int16)

def transcribe_worker():
    global audio_history
    print("\n--- Lenition Parser Active (IPA Mode) ---")
    

    while True:
        new_data = []
        while not data_queue.empty():
            new_data.append(data_queue.get())
        
        if new_data:
            incoming = np.concatenate(new_data)
            if len(incoming) > MAX_SAMPLES:
                incoming = incoming[-MAX_SAMPLES:]
            
            num_new = len(incoming)
            audio_history = np.roll(audio_history, -num_new)
            audio_history[-num_new:] = incoming

            audio_float = audio_history.astype(np.float32) / 32768.0
            
            # 1. Whisper Intent
            segments, _ = whisper_model.transcribe(audio_float, language="en", beam_size=1)
            canonical = " ".join([s.text for s in segments]).strip()

            # 2. Phonetic Reality (IPA)
            inputs = processor(audio_float, return_tensors="pt", sampling_rate=16000).input_values.to("cpu")
            with torch.no_grad():
                logits = phonetic_model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            ipa_text = processor.batch_decode(predicted_ids)[0]



            if len(canonical) > 2:
                print(f"\n{'-'*40}")
                
                ipa_text = processor.batch_decode(predicted_ids)[0]
                words = canonical.split()
                display_intent_list = [w.upper() for w in words]
                display_ipa = ipa_text
                
                flap_idx = ipa_text.find("ɾ")
                
                if flap_idx != -1:
                    # 1. Calculate relative position of the flap in the IPA string
                    # This acts as a proxy for 'time' within the 3-second window
                    relative_pos = flap_idx / len(ipa_text)
                    
                    # 2. Map that position to the corresponding word in the Intent
                    word_map_idx = int(relative_pos * len(words))
                    word_map_idx = min(word_map_idx, len(words) - 1)
                    
                    # 3. Highlight only that specific word if it contains a T
                    target_word = display_intent_list[word_map_idx]
                    if 'T' in target_word:
                        display_intent_list[word_map_idx] = f"{YELLOW}{BOLD}{target_word}{ENDC}"
                    
                    # 4. Highlight IPA segment (2 chars to each side)
                    start = max(0, flap_idx - 2)
                    end = min(len(ipa_text), flap_idx + 3)
                    target_segment = ipa_text[start:end]
                    display_ipa = ipa_text[:start] + f"{YELLOW}{BOLD}{target_segment}{ENDC}" + ipa_text[end:]

                # Reconstruct the Intent string
                display_intent = " ".join(display_intent_list)

                print(f"INTENT: {display_intent}")
                print(f"RAW IPA: {display_ipa}")
                
                if "TT" in canonical.upper() and flap_idx != -1:
                    print(">>> DETECTED: Intervocalic Flapping")

        time.sleep(0.5)

def audio_callback(in_data, frame_count, time_info, status):
    data_queue.put(np.frombuffer(in_data, dtype=np.int16))
    return (None, pyaudio.paContinue)

p = pyaudio.PyAudio()
try:
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                    input_device_index=DEVICE_INDEX, frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    t = threading.Thread(target=transcribe_worker, daemon=True)
    t.start()

    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    if 'stream' in locals():
        stream.stop_stream()
        stream.close()
    p.terminate()
