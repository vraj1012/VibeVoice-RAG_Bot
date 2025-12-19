import sounddevice as sd
import numpy as np
import threading
import time
import asyncio
import websockets
import urllib.parse
import config

class AudioPlayer:
    def __init__(self):
        self.stream = sd.OutputStream(samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, dtype='int16', callback=self.callback)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.is_speaking = False
        self.playback_start = 0
        self.stream.start()

    def callback(self, outdata, frames, time_info, status):
        bytes_needed = frames * 2
        with self.lock:
            if len(self.buffer) == 0:
                outdata.fill(0)
                self.is_speaking = False
            else:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.playback_start = time.time()
                
                chunk = self.buffer[:bytes_needed]
                del self.buffer[:bytes_needed]
                if len(chunk) < bytes_needed:
                    chunk += bytes(bytes_needed - len(chunk))
                outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)

    def add_data(self, data):
        with self.lock:
            if len(self.buffer) == 0:
                self.playback_start = time.time()
            self.buffer.extend(data)
    
    def stop_immediately(self):
        duration = 0
        if self.is_speaking:
            duration = time.time() - self.playback_start
        
        with self.lock: self.buffer.clear()
        self.is_speaking = False
        return duration

# --- UPDATE 1: Accept audio_buffer argument ---
async def stream_audio_from_server(text, audio_player, event, audio_buffer):
    if not text: return
    clean_text = text.replace("*", "").strip()
    
    params = { 
        "text": clean_text, "cfg": "1.5", "steps": "6", 
        "voice": config.VOICE_ID, "speed": "1.0" 
    }
    url = f"{config.VOICE_SERVER_URL}?{urllib.parse.urlencode(params)}"

    try:
        async with websockets.connect(url) as websocket:
            async for message in websocket:
                if event.is_set(): break 
                if isinstance(message, str): continue
                
                audio_player.add_data(message)
                
                # --- UPDATE 2: Save Bot Audio to Buffer ---
                if audio_buffer is not None:
                    audio_buffer.extend(message)
                    
    except: pass

# --- UPDATE 3: Pass it down ---
def speak(text, audio_player, event, audio_buffer=None):
    event.clear()
    asyncio.run(stream_audio_from_server(text, audio_player, event, audio_buffer))