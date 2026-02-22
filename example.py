import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
from faster_whisper import WhisperModel

# 1. Initialize Whisper (The Ears)
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. Initialize Piper (The Mouth)
voice = PiperVoice.load("en_US-amy-low.onnx")

def speak(text):
    print(f">>> Piper is saying: {text}")

    for chunk in voice.synthesize(text):
        # Use the pre-computed float array from the AudioChunk object
        audio_float32 = chunk.audio_float_array

        # Play it back using the sample rate from the chunk itself
        # (This ensures it matches the audio data perfectly)
        sd.play(audio_float32, samplerate=chunk.sample_rate)
        sd.wait()


def run_loop():
    fs = 16000
    seconds = 5

    print("\n[READY] Speak now (5 second window)...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    # Transcribe the audio
    segments, _ = stt_model.transcribe(recording.flatten())
    user_text = " ".join([s.text for s in segments]).strip()

    if user_text:
        print(f"User: {user_text}")
        response = f"You said {user_text}."
        speak(response)
    else:
        print("...Silence...")

if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\nExiting...")
