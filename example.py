import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# 1. Load the model (tiny is fast, base is more accurate)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def record_and_transcribe():
    fs = 16000  # Sample rate
    seconds = 5  # Duration

    print(f">>> Recording for {seconds} seconds...")
    # Record audio as a numpy array
    audio_data = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished

    # Flatten the array for Whisper
    audio_data = audio_data.flatten()

    print(">>> Transcribing...")
    segments, info = model.transcribe(audio_data, beam_size=5)

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    try:
        record_and_transcribe()
    except Exception as e:
        print(f"Caught Error: {e}")
