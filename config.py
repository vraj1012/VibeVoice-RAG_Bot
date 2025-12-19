import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 🎛️  MASTER SWITCH
# Options: "local" (Llama 3 8B) OR "cloud" (Gemini Flash)
# ==========================================
AI_MODE = "local" 
# AI_MODE = "cloud" 

# --- MODEL CONFIGURATIONS ---
# Local Setting (Best Open Source)
LOCAL_MODEL_NAME = "llama3:8b-instruct-q4_K_M"

# Cloud Setting (Best Paid API)
CLOUD_API_KEY = os.getenv("GEMINI_API_KEY")
CLOUD_MODEL_NAME = "gemini-2.5-flash"

# --- VOICE SETTINGS ---
VOICE_SERVER_URL = "ws://localhost:3000/stream"
VOICE_ID = "en-Emma_woman"
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
MIC_THRESHOLD = 300
WORDS_PER_SECOND = 2.5 # Used for smart resume calculation

# --- PHRASES ---
GREETING_PHRASES = ["hello", "hi", "hey", "good morning", "good afternoon"]
CLOSING_PHRASES = ["thank", "thanks", "bye", "goodbye"]
BACKCHANNEL_LIST = ["ok", "okay", "hmm", "uh-huh", "right", "yeah", "yep", "go on", "got it", "i see", "understood"]