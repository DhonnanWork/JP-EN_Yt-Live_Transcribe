# Live YouTube JP-to-EN Transcriber & Translator

Automatic JP -> EN live translation with `faster-whisper` (medium) & `Helsinki-NLP-opus-mt-ja-en` for transcription and translation.

It captures audio directly from YouTube Live streams (or standard videos) in real-time using `yt-dlp` and `ffmpeg`, processes the audio in small chunks, formats the Japanese text with Kanji, perfectly aligned Hiragana (via `pykakasi`), and outputs contextual English translations. 

**Tested & Built on:** Colorful Evol P15 / i5-13420H / RTX 5050 Laptop GPU (8GB VRAM) running Zorin OS / Linux.

## Features
- **Hardware Accelerated:** Uses CUDA and Float16 compute for near-instant transcription.
- **Contextual Translation:** Remembers previous sentences to improve translation accuracy.
- **VAD Filtering:** Uses Voice Activity Detection to prevent Whisper hallucination loops during silence.
- **Smart GUI:** Built-in Tkinter interface with "Clear Screen" and "Reset AI Context" buttons.
- **System Checks:** Automatically verifies OS compatibility, `.env` Hugging Face tokens, and VRAM requirements on startup.

## Prerequisites
You must have the following installed on your system:
- **Python 3.10+**
- **FFmpeg** (`sudo apt install ffmpeg` on Linux, or added to PATH on Windows)
- **Node.js** (`sudo apt install nodejs` - required by `yt-dlp` to bypass YouTube signature algorithms)
- **NVIDIA GPU** with CUDA installed (8GB+ VRAM Highly Recommended).

*(Disclaimer for Windows Users: This script uses UNIX-style piping for FFmpeg. Ensure FFmpeg is correctly installed in your system PATH, otherwise the script will fail to capture audio).*

````markdown
# Installation & Setup

## 1. Clone the repository
```bash
git clone https://github.com/DhonnanWork/JP-EN_Yt-Live_Transcribe
cd JP-EN_Yt-Live_Transcribe
````

## 2. Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Setup Hugging Face Token

1. Create a file named `.env` in the root directory.
2. Add your Hugging Face read token inside:

```env
HF_TOKEN="hf_your_token_here"
```
