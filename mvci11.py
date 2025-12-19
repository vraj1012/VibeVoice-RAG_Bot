import os
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import google.generativeai as genai 
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import urllib.parse
import threading
import time
import speech_recognition as sr
import wave
from datetime import datetime
import queue 
import re 

from faster_whisper import WhisperModel

# --- 1. CONFIGURATION ---
VOICE_SERVER_URL = "ws://localhost:3000/stream"
SAMPLE_RATE = 24000 
SAMPLE_WIDTH = 2     
CHANNELS = 1         

# VOICE
VOICE_ID = "en-Emma_woman" 

# AUDIO SETTINGS
CFG_SCALE = "1.5" 
INFERENCE_STEPS = "6"  
DEFAULT_SPEED = "1.0"
MIC_THRESHOLD = 300 
WORDS_PER_SECOND = 2.5  # Average speaking rate

# SMART LISTS
GREETING_PHRASES = ["hello", "hi", "hey", "good morning", "good afternoon"]
CLOSING_PHRASES = ["thank", "thanks", "bye", "goodbye"]
# Expanded list to catch "got it", "ok", etc.
BACKCHANNEL_LIST = ["ok", "okay", "hmm", "uh-huh", "right", "yeah", "yep", "go on", "got it", "i see", "understood"]

load_dotenv()

# --- 2. SETUP ---
print("⏳ Loading System...")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

embedder = SentenceTransformer('all-MiniLM-L6-v2')
import chromadb
chroma_client = chromadb.PersistentClient(path="my_rag_db")
collection = chroma_client.get_or_create_collection(name="avocado_knowledge")

print("⏳ Loading Whisper (base.en)...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

print(f"✅ System Ready! ({VOICE_ID})")
print("🎧 IMPORTANT: HEADPHONES REQUIRED")

# --- 3. STATE ---
conversation_memory = [] 
last_bot_response_text = "" 
# NEW: Store what was left unsaid
remaining_text_buffer = "" 

full_audio_buffer = bytearray()
user_input_queue = queue.Queue()       
interruption_event = threading.Event() 

def log_interaction(role, text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    conversation_memory.append(f"{role}: {text}")
    print(f"[{timestamp}] {role}: {text}")
    
    if role == "Bot":
        global last_bot_response_text
        last_bot_response_text = text

def save_data():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(f"transcript_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_memory))
    try:
        with wave.open(f"recording_{timestamp}.wav", 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(full_audio_buffer)
    except: pass

def clean_llm_response(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'^(Bot|User|Maya|System):', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*.*?\*', '', text)
    text = re.sub(r'IMPORTANT:.*', '', text, flags=re.IGNORECASE)
    return text.strip()

# --- 4. AUDIO ENGINE (WITH STOPWATCH) ---
class AudioPlayer:
    def __init__(self):
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=self.callback)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.is_speaking = False
        
        # TIMING STATE
        self.playback_start = 0
        self.total_bytes_played = 0
        
        self.stream.start()

    def callback(self, outdata, frames, time, status):
        if interruption_event.is_set():
            outdata.fill(0)
            with self.lock: self.buffer.clear()
            return

        bytes_needed = frames * 2
        with self.lock:
            if len(self.buffer) == 0:
                outdata.fill(0)
                self.is_speaking = False
            else:
                if not self.is_speaking:
                    # START TIMER when sound actually starts
                    self.is_speaking = True
                    self.playback_start = time.currentTime
                
                chunk = self.buffer[:bytes_needed]
                bytes_to_play = len(chunk)
                del self.buffer[:bytes_needed]
                
                if bytes_to_play < bytes_needed:
                    chunk += bytes(bytes_needed - bytes_to_play)
                
                outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
                
                # We could count bytes, but time is easier for "words"
                pass

    def add_data(self, data):
        with self.lock: 
            if len(self.buffer) == 0:
                self.playback_start = time.time() # Reset timer on new sentence
            self.buffer.extend(data)
    
    def stop_immediately(self):
        # CALCULATE HOW LONG WE SPOKE
        duration = 0
        if self.is_speaking:
            duration = time.time() - self.playback_start
        
        interruption_event.set()
        with self.lock: self.buffer.clear()
        self.is_speaking = False
        
        return duration

audio_player = AudioPlayer()

# --- 5. BACKGROUND LISTENER ---
recognizer = sr.Recognizer()
MIC_DEVICE_INDEX = 1  
mic = sr.Microphone(device_index=MIC_DEVICE_INDEX)

def background_listener_callback(recognizer, audio):
    try:
        audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, info = whisper_model.transcribe(audio_np, beam_size=5, vad_filter=True)
        text = " ".join([segment.text for segment in segments]).strip()
        
        if not text: return 

        was_interrupted = False
        duration_spoken = 0
        
        if audio_player.is_speaking:
            print(f"\n🚧 Interruption Detected! ('{text}')")
            # GET DURATION BEFORE STOPPING
            duration_spoken = audio_player.stop_immediately()
            was_interrupted = True 

        print(f"\n🗣️ User detected: {text}")
        
        raw_bytes = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=SAMPLE_WIDTH)
        full_audio_buffer.extend(raw_bytes)
        
        # Pass duration to main loop
        user_input_queue.put((text, was_interrupted, duration_spoken)) 
        
    except Exception as e:
        print(f"⚠️ Listener Error: {e}")

print(f"👂 Background Listener Active (Device {MIC_DEVICE_INDEX})...")

with mic as source:
    recognizer.energy_threshold = MIC_THRESHOLD  
    recognizer.dynamic_energy_threshold = False 
    recognizer.pause_threshold = 1.2 

stop_listening = recognizer.listen_in_background(mic, background_listener_callback, phrase_time_limit=None)

# --- 6. VOICE CLIENT ---
async def stream_audio_from_server(text):
    if not text: return
    interruption_event.clear()
    
    clean_text = text.replace("*", "").strip()
    
    params = { 
        "text": clean_text, 
        "cfg": CFG_SCALE, 
        "steps": INFERENCE_STEPS, 
        "voice": VOICE_ID, 
        "speed": DEFAULT_SPEED 
    }
    url = f"{VOICE_SERVER_URL}?{urllib.parse.urlencode(params)}"

    try:
        async with websockets.connect(url) as websocket:
            async for message in websocket:
                if interruption_event.is_set(): break 
                if isinstance(message, str): continue
                audio_player.add_data(message)
                full_audio_buffer.extend(message)
    except: pass

def speak(text):
    interruption_event.clear() 
    asyncio.run(stream_audio_from_server(text))

# --- 7. BRAIN (GEMINI FLASH WITH SMART RESUME) ---
def get_interactive_response(user_text, override_instruction=None):
    
    clean_user_text = user_text.lower().strip(" .!,?")
    user_words = clean_user_text.split()
    
    is_social_only = False
    if any(p in clean_user_text for p in CLOSING_PHRASES) and len(user_words) < 6:
        is_social_only = True

    if is_social_only:
        context = "" 
        recent_history = ""
        print("🔕 Social Mode: Context Nuke")
    else:
        query_embedding = embedder.encode(user_text).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        context = "\n".join(results['documents'][0]) if results['documents'][0] else ""
        recent_history = "\n".join(conversation_memory[-6:]) 

    # --- INSTRUCTION INJECTION ---
    system_instruction = ""
    
    if any(g in clean_user_text for g in GREETING_PHRASES) and len(user_words) < 5 and not override_instruction:
        system_instruction = "IMPORTANT: User is greeting. Reply ONLY: 'Hi there! I'm Maya. How can I help you with avocados?'"
    
    elif any(t in clean_user_text for t in CLOSING_PHRASES) and len(user_words) < 5 and not override_instruction:
        system_instruction = "IMPORTANT: User said thanks. Reply ONLY: 'You're welcome!'"
    
    elif override_instruction:
        system_instruction = f"IMPORTANT: {override_instruction}"

    # --- PROMPT ---
    system_prompt = f"""
    You are 'Maya', a friendly interactive guide for an Avocado Website.
    
    CONTEXT DATA:
    {context}
    
    CHAT HISTORY:
    {recent_history}
    
    {system_instruction}
    
    YOUR STYLE:
    1. **Conversational:** Use natural language like "Honestly," "You know," or "Here's the thing."
    2. **Stay on Topic:** Do NOT make up stories about a farm or a mom. Stick to the avocado facts provided.
    3. **No Hallucinations:** Do NOT say "Hello" or "You're welcome" unless the user explicitly greeted you or thanked you first.
    4. **Smart Resume:** If instructed to resume, do NOT repeat the first part. Start exactly where you left off with a transition like "So, as I was saying..."
    5. **Short:** Keep answers under 40 words.
    
    User Input: {user_text}
    """

    try:
        response = gemini_model.generate_content(system_prompt)
        return clean_llm_response(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Hmm, I'm having trouble connecting. Say again?"

# --- 8. MAIN LOOP ---
if __name__ == "__main__":
    print("\n🥑 MAYA SMART RESUME EDITION")
    print("---------------------------------------------")
    
    print("🔥 Warming up models...")
    _ = whisper_model.transcribe(np.zeros(16000).astype(np.float32), beam_size=1) 
    
    start_msg = "Hey! Maya here."
    log_interaction("Bot", start_msg)
    speak(start_msg)
    
    try:
        while True:
            try:
                input_data = user_input_queue.get(timeout=0.1)
                
                if isinstance(input_data, tuple):
                    user_text = input_data[0]
                    was_interrupted = input_data[1]
                    duration_spoken = input_data[2] if len(input_data) > 2 else 0
                else:
                    user_text = input_data
                    was_interrupted = False
                    duration_spoken = 0
                
                log_interaction("User", user_text)
                
                if any(x in user_text.lower() for x in ["bye", "exit", "quit"]):
                    end_msg = "Alright, catch you later!"
                    log_interaction("Bot", end_msg)
                    speak(end_msg)
                    break
                
                instruction = None
                clean_text = user_text.lower().strip(" .!,")
                
                if was_interrupted:
                    # 1. CALCULATE WHAT WAS SPOKEN
                    words_estimated = int(duration_spoken * WORDS_PER_SECOND)
                    full_words = last_bot_response_text.split()
                    
                    # 2. SLICE TEXT
                    if words_estimated < len(full_words):
                        spoken_part = " ".join(full_words[:words_estimated])
                        remaining_part = " ".join(full_words[words_estimated:])
                        
                        # Store for "Smart Resume"
                        remaining_text_buffer = remaining_part
                        print(f"DEBUG: Spoken: '{spoken_part}' | Left: '{remaining_part}'")
                    else:
                        remaining_text_buffer = ""

                    # 3. DECIDE INTENT
                    if "no" in clean_text.split()[:2] or "wait" in clean_text.split()[:2]:
                        instruction = f"User is correcting you. Apologize and ask for clarification."
                    
                    elif any(phrase in clean_text for phrase in BACKCHANNEL_LIST):
                         # SMART RESUME LOGIC
                         if remaining_text_buffer:
                             instruction = f"User agreed ('{user_text}'). You were interrupted. You already said: '{spoken_part}'. NOW, only say the rest: '{remaining_part}'. Use a smooth transition like 'So, anyway...' or 'Right, and...'"
                         else:
                             instruction = f"User agreed. Continue previous point: '{last_bot_response_text}'."
                    
                    elif len(clean_text.split()) < 3:
                         instruction = "User mumbled. Ignore context. Say 'Oh, sorry, go ahead?'"
                    else:
                         instruction = f"User interrupted with: '{user_text}'. Answer this new question."
                
                answer = get_interactive_response(user_text, override_instruction=instruction)
                
                log_interaction("Bot", answer)
                speak(answer)
                
            except queue.Empty:
                continue 
            
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        print("\nDisconnected.")
    finally:
        save_data()