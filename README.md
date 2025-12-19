# 🥑 Vibe Voice Assistant (Maya)

Maya is a next-generation voice AI assistant powered by the **Vibe Voice** TTS engine. Our project uses **Local LLM** (Llama 3 8B) or **Cloud Brain** (Gemini 2.5 Flash).

## ✨ Key Features

* **🎙️ Vibe Voice TTS:** Uses the custom `vibevoice` streaming engine for ultra-realistic, low-latency speech synthesis.
* **🧠 LLM's used:**
    * **Local Mode:** **Llama 3 8B** via Ollama for complete privacy and offline capability.
    * **Cloud Mode:** **Gemini 2.5 Flash** for better response, reasoning and expanded knowledge.
* **📚 Context-Aware RAG:** Remembers conversation history and retrieves specific agricultural knowledge from a local `ChromaDB` vector store.
* **⚡ Smart Interruptions (Barge-In):** The bot listens while speaking. If you interrupt, it stops instantly and handles the new context gracefully.
* **⏱️ Smart Resume:** If cut off mid-sentence, the bot calculates exactly what was spoken and can resume the thought naturally.

## 🛠️ Tech Stack & Environment

* **Environment:** Conda (`vvoice`)
* **TTS Engine:** Vibe Voice (WebSocket Stream)
* **LLM Backend:** Ollama & Google Gemini
* **Speech Recognition:** Faster-Whisper (Int8 Quantized)
* **Audio Backend:** PortAudio / PyAudio / SoundDevice

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have **Anaconda**  installed.

### 2. Environment Setup
We use a specific environment named `vvoice`.

```bash
# 1. Create the environment
conda create -n vvoice python=3.10

# 2. Activate the environment
conda activate vvoice

# 3. Install Audio Drivers (Required for Microphone)
# For Windows, PyAudio usually installs via pip, but if it fails:
# conda install pyaudio

```

### 3. Install GPU Support (Critical)

You must install the CUDA-enabled versions of PyTorch to run the AI fast. Run this exact command:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

### 4. Install Vibe Voice

Navigate to your VibeVoice directory and install it in editable mode:

```bash
cd path/to/VibeVoice
pip install -e .

```

### 5. Install Project Dependencies

Now install the rest of the libraries for Maya:

```bash
pip install -r requirements.txt

```

### 6. Setup LLMs

**Local (Ollama):**

1. Download [Ollama](https://ollama.com/).
2. Pull the 8B model:
```bash
ollama pull llama3:8b-instruct-q4_K_M

```



**Cloud (Gemini):**

1. Get an API Key from [Google AI Studio](https://aistudio.google.com/).
2. Create a `.env` file in the project root:
```ini
GEMINI_API_KEY=your_actual_api_key_here

```



## ⚙️ Configuration

You can control the entire system using **`config.py`**.

**Switching Intelligence:**

```python
# config.py

# Set to "local" for Llama 3 (Offline)
# Set to "cloud" for Gemini Flash (Fastest)
AI_MODE = "local" 

```

## 🏃‍♂️ Usage

1. **Start the TTS Server:**
You must start the voice server before running the bot. Open a **separate terminal**, activate your environment, and run the demo server:

```bash
conda activate vvoice
cd VibeVoice/demo
python vibevoice_realtime_demo.py --port 3000

2. **Run the Bot:**
```bash
python main.py

```



## 📂 Project Structure

```
Avocado_AI_Bot/
│
├── config.py              # 🎛️ Master Control (Model Switching, Audio Settings)
├── main.py                # 🏁 Entry Point
├── requirements.txt       # 📦 Dependencies
├── README.md              # 📄 This file
│
└── src/
    ├── audio.py           # 🔊 Vibe Voice WebSocket Client & Audio Player
    ├── ears.py            # 👂 Faster-Whisper Implementation
    ├── brain.py           # 🧠 Logic for Gemini & Llama
    ├── memory.py          # 📚 ChromaDB RAG Manager
    └── utils.py           # 🛠️ Helpers

```

```

```

## 🐛 Troubleshooting

* **"Ollama connection refused":** Make sure the Ollama app is running in the background (check your system tray).
* **"WebSocket Error":** This means the Vibe Voice server isn't running. Ensure it is active on `ws://localhost:3000`.
* **Microphone Issues:** Check the console output for "Background Listener Active". If it crashes instantly, check your `device_index` in `config.py`.
* **"Torch not compiled with CUDA enabled":** You likely installed the wrong PyTorch version. Run the "Install GPU Support" command in the Installation section again.

## 📄 License

[MIT License](LICENSE)