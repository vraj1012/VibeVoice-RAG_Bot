import re
from datetime import datetime

def log_interaction(role, text, conversation_memory):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"{role}: {text}"
    conversation_memory.append(entry)
    print(f"[{timestamp}] {role}: {text}")

def clean_llm_response(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'^(Bot|User|Maya|System):', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*.*?\*', '', text)
    text = re.sub(r'IMPORTANT:.*', '', text, flags=re.IGNORECASE)
    return text.strip()