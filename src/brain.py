import google.generativeai as genai
import ollama
import config
from src.utils import clean_llm_response

class BrainEngine:
    def __init__(self):
        self.mode = config.AI_MODE
        print(f"🧠 Brain Mode: {self.mode.upper()}")
        
        if self.mode == "cloud":
            genai.configure(api_key=config.CLOUD_API_KEY)
            self.model = genai.GenerativeModel(config.CLOUD_MODEL_NAME)
        else:
            self.model_name = config.LOCAL_MODEL_NAME

    def generate(self, user_text, context, history, instruction=None):
        system_instruction = f"IMPORTANT: {instruction}" if instruction else ""
        
        prompt = f"""
        You are 'Maya', a friendly interactive guide for an Avocado Website.
        
        CONTEXT DATA:
        {context}
        
        CHAT HISTORY:
        {history}
        
        {system_instruction}
        
        YOUR STYLE:
        1. **Conversational:** Use natural language like "Honestly," "You know," or "Here's the thing."
        2. **Stay on Topic:** Stick to the avocado facts provided.
        3. **No Hallucinations:** Do NOT say "Hello" or "You're welcome" unless user initiates.
        4. **Smart Resume:** If instructed to resume, do NOT repeat the first part.
        5. **Short:** Keep answers under 40 words.
        
        User Input: {user_text}
        """

        try:
            if self.mode == "cloud":
                response = self.model.generate_content(prompt)
                return clean_llm_response(response.text)
            else:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'system', 'content': prompt},
                              {'role': 'user', 'content': user_text}],
                    options={'temperature': 0.7}
                )
                return clean_llm_response(response['message']['content'])
        except Exception as e:
            print(f"Brain Error: {e}")
            return "Hmm, I'm having trouble thinking right now."