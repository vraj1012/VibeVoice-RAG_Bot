import speech_recognition as sr
import numpy as np
from faster_whisper import WhisperModel
import config

class EarEngine:
    def __init__(self, device_index=1):
        print("⏳ Loading Whisper Ears...")
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(device_index=device_index)
        self.recognizer.energy_threshold = config.MIC_THRESHOLD
        self.recognizer.dynamic_energy_threshold = False

    def listen_background(self, callback):
        return self.recognizer.listen_in_background(self.mic, callback, phrase_time_limit=None)

    def transcribe_audio(self, audio_data):
        try:
            raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = self.model.transcribe(audio_np, beam_size=5, vad_filter=True)
            return " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            print(f"Ear Error: {e}")
            return ""

    # --- NEW: WARMUP FUNCTION ---
    def warmup(self):
        """Runs a dummy transcription to load the model into memory."""
        print("🔥 Warming up Whisper model...")
        # Create 1 second of silence (32000 bytes = 16000Hz * 2 bytes/sample)
        dummy_audio = sr.AudioData(b'\0'*32000, 16000, 2)
        self.transcribe_audio(dummy_audio)