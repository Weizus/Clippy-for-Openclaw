import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
from faster_whisper import WhisperModel

# 1. Initialize Whisper (The Ears)
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# 2. Initialize Piper (The Mouth)
# Point this to the file you just downloaded
voice = PiperVoice.load("en_US-amy-low.onnx")

def speak(text):
    print(f">>> Piper is saying: {text}")

    # Piper's synthesize returns a generator of AudioChunk objects
    for chunk in voice.synthesize(text):
        # Access the raw bytes inside the AudioChunk object
        audio_bytes = chunk.audio

        # Convert bytes to 16-bit integers
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 for sounddevice (-1.0 to 1.0)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Use the sample rate defined in the voice's own config (usually 22050)
        # This is safer than hardcoding it!
        sd.play(audio_float32, samplerate=voice.config.sample_rate)
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
        response = f"You said {user_text}. That sounds interesting."
        speak(response)
    else:
        print("...Silence...")

if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\nExiting...")
