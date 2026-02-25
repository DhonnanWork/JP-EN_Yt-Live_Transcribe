***

# 🎙️ AI Live Transcriber & Translator (RTX Optimized)

A local, GPU-accelerated application that transcribes and translates audio in real-time. Supports **YouTube Livestreams**, **Local Audio/Video files**, and **Microphone Input**.

Designed for **language learning** (specifically Japanese) with a multi-layer breakdown, but works universally for many languages using state-of-the-art AI models.

## ✨ Features

*   **Multi-Source Input:**
    *   🔴 **YouTube Live & VOD:** Stream directly from YouTube without downloading video (Audio-only processing).
    *   🎤 **Microphone:** Live translation of your voice or system audio.
    *   📁 **Local Files:** Transcribe `.mp4`, `.wav`, `.mp3`, etc.
*   **Dual Translation Engines:**
    *   🚀 **Helsinki-NLP:** Extremely fast, language-pair specific models (Best for EN↔JP, EN↔ID, etc.).
    *   🌐 **NLLB-200 (Facebook):** Universal fallback model supporting 200+ languages with high accuracy.
*   **Japanese Learning Mode:**
    *   Generates a 4-layer breakdown for students:
        1.  **Kanji** (Original)
        2.  **Hiragana** (Reading)
        3.  **Glossary** (Literal word-by-word meaning)
        4.  **Translation** (Full coherent sentence)
*   **Optimized for Consumer GPUs (RTX 30xx/40xx/50xx):**
    *   Uses `float16` precision to cut VRAM usage in half.
    *   **Smart Memory Management:** Automatically loads/unloads models and clears Garbage Collection to prevent Out-Of-Memory crashes.
    *   **CPU Fallback:** Automatically detects low VRAM and switches to CPU if necessary.

## 🛠️ Prerequisites

Before running the Python script, ensure your system has the following installed:

### 1. System Dependencies (Linux/Debian/Ubuntu/Zorin)
You need **FFmpeg** for audio processing and **NodeJS** for the latest YouTube anti-bot bypass.

```bash
sudo apt update
sudo apt install ffmpeg nodejs
```

### 2. NVIDIA Drivers & CUDA
Ensure you have the proprietary NVIDIA drivers installed. You can check with:
```bash
nvidia-smi
```

## 📦 Installation

1.  **Clone the repository (or create folder):**
    ```bash
    mkdir Transcribe_Project
    cd Transcribe_Project
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Libraries:**
    Create a file named `requirements.txt` with the following content:
    ```text
    torch
    torchaudio
    --extra-index-url https://download.pytorch.org/whl/cu118
    faster-whisper
    transformers
    sacremoses
    sentencepiece
    sounddevice
    numpy
    yt-dlp
    pykakasi
    python-dotenv
    requests
    tk
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create .env file (Optional):**
    If you plan to use HuggingFace private models or specific yt-dlp configurations.
    ```bash
    touch .env
    ```

## 🚀 Usage

1.  **Run the Application:**
    ```bash
    python3 main.py
    ```

2.  **Configuration Window:**
    When the app starts, a "New Input Configuration" window appears.

    *   **1. Source:** Select YouTube, Local File, or Mic.
    *   **2. Input:** Paste the URL or File Path.
    *   **3. Model:**
        *   *Helsinki-NLP:* Faster, recommended for specific pairs (e.g., JA->EN).
        *   *NLLB-200:* Slower but supports almost any language combination.
    *   **4. Layers (Japanese Only):** Toggle specific output lines (Kanji, Hiragana, Word Meaning, Sentence).

3.  **Control:**
    *   Click **RUN** to start.
    *   Click **STOP** to end capture (keeps AI loaded).
    *   Click **New Input** to change source/language (Reloads AI context).

## 🇯🇵 Japanese Learning Mode Output

When translating Japanese to English with all layers enabled, the output looks like this:

```text
[02:14]
週間       の      次       は      １年         行きましょう   
しゅうかん  の      つぎ      は      いちねん      いきましょう    
weeks      's      next     topic    one year     let's go      
TRANS: Let's go for one year after the weeks.
----------------------------------------------------------------------
```

## 🔧 Troubleshooting

**1. `[YT-DLP Error] No supported JavaScript runtime found`**
*   **Fix:** You are missing NodeJS. YouTube requires it to calculate stream signatures.
*   Run: `sudo apt install nodejs`

**2. `CUDA failed with error out of memory`**
*   The app tries to prevent this automatically. However, if it happens:
    *   Close web browsers (Chrome/Firefox eat VRAM).
    *   The app will automatically try to fallback to CPU on the next run if VRAM is < 3.5GB.

**3. Timestamps are looping (0:00 - 0:08)**
*   This is fixed in the latest version.
    *   **Static Files:** Timestamps represent the actual video time.
    *   **Livestream/Mic:** Timestamps represent elapsed time since you clicked "Run".

**4. YouTube blocking connections (HTTP 429)**
*   If you download too much, YouTube soft-bans your IP.
*   **Fix:** Export your cookies from your browser (using a "Get cookies.txt" extension) to a file named `youtube_cookies.txt`.
*   Uncomment the cookie section in `audio_capture.py`.

## 📜 License
This project uses open-source models:
*   **Whisper** by OpenAI (MIT)
*   **NLLB-200** by Meta AI (MIT)
*   **Helsinki-NLP** models (Apache 2.0)