import queue
import threading
import time
import wave
from datetime import datetime

import config
from src.memory import MemoryEngine
from src.brain import BrainEngine
from src.audio import AudioPlayer, speak
from src.ears import EarEngine
from src.utils import log_interaction

# --- STATE ---
user_input_queue = queue.Queue()
interruption_event = threading.Event()
full_audio_buffer = bytearray()  # <--- This holds the recording
last_bot_response_text = ""

# --- INITIALIZE MODULES ---
memory = MemoryEngine()
brain = BrainEngine()
mouth = AudioPlayer()
ears = EarEngine(device_index=1)

# --- CALLBACK FOR EARS ---
def listener_callback(recognizer, audio):
    text = ears.transcribe_audio(audio)
    if not text: return

    was_interrupted = False
    duration_spoken = 0
    
    if mouth.is_speaking:
        print(f"\n🚧 Interruption Detected! ('{text}')")
        duration_spoken = mouth.stop_immediately()
        interruption_event.set()
        was_interrupted = True 

    print(f"\n🗣️ User detected: {text}")
    
    # Save USER audio to buffer
    raw_bytes = audio.get_raw_data(convert_rate=config.SAMPLE_RATE, convert_width=config.SAMPLE_WIDTH)
    full_audio_buffer.extend(raw_bytes)
    
    user_input_queue.put((text, was_interrupted, duration_spoken))

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"\n🥑 MAYA BOT ONLINE [{config.AI_MODE.upper()} MODE]")
    print("---------------------------------------------")
    
    ears.warmup()
    
    start_msg = "Hey! Maya here."
    log_interaction("Bot", start_msg, memory.history)
    
    # FIX 1: Pass full_audio_buffer to capture Bot's voice
    speak(start_msg, mouth, interruption_event, full_audio_buffer)
    
    stop_listening = ears.listen_background(listener_callback)
    
    try:
        while True:
            try:
                input_data = user_input_queue.get(timeout=0.1)
                
                user_text, was_interrupted, duration_spoken = input_data
                
                log_interaction("User", user_text, memory.history)
                
                if any(x in user_text.lower() for x in ["bye", "exit", "quit"]):
                    end_msg = "Alright, catch you later!"
                    log_interaction("Bot", end_msg, memory.history)
                    speak(end_msg, mouth, interruption_event, full_audio_buffer)
                    break
                
                instruction = None
                clean_text = user_text.lower().strip(" .!,")
                
                if was_interrupted:
                    words_estimated = int(duration_spoken * config.WORDS_PER_SECOND)
                    full_words = last_bot_response_text.split()
                    
                    spoken_part = ""
                    remaining_part = ""
                    
                    if words_estimated < len(full_words):
                        spoken_part = " ".join(full_words[:words_estimated])
                        remaining_part = " ".join(full_words[words_estimated:])
                    
                    if "no" in clean_text.split()[:2] or "wait" in clean_text.split()[:2]:
                        instruction = "User is correcting you. Apologize."
                    elif any(p in clean_text for p in config.BACKCHANNEL_LIST):
                        if remaining_part:
                            instruction = f"User agreed. You were cut off. You said: '{spoken_part}'. Finish saying: '{remaining_part}'."
                        else:
                            instruction = f"User agreed. Continue previous point."
                    elif len(clean_text.split()) < 3:
                        instruction = "User mumbled. Ignore."
                    else:
                        instruction = f"User interrupted. Answer new question: {user_text}"

                context = memory.get_context(user_text)
                history = memory.get_recent_history()
                
                answer = brain.generate(user_text, context, history, instruction)
                
                last_bot_response_text = answer
                log_interaction("Bot", answer, memory.history)
                
                # FIX 2: Pass full_audio_buffer here too
                speak(answer, mouth, interruption_event, full_audio_buffer)
                
            except queue.Empty:
                continue 
            
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        print("\nDisconnected.")
    finally:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Save Transcript
        with open(f"transcript_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(memory.history))
            
        # FIX 3: Restore the WAV file saving logic!
        try:
            with wave.open(f"recording_{timestamp}.wav", 'wb') as wf:
                wf.setnchannels(config.CHANNELS)
                wf.setsampwidth(config.SAMPLE_WIDTH)
                wf.setframerate(config.SAMPLE_RATE)
                wf.writeframes(full_audio_buffer)
            print(f"\n💾 Saved Session: recording_{timestamp}.wav")
        except Exception as e:
            print(f"Error saving audio: {e}")