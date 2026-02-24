import os
import subprocess
import numpy as np
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
import platform
from dotenv import load_dotenv

import torch
from faster_whisper import WhisperModel
import pykakasi
from transformers import MarianMTModel, MarianTokenizer

load_dotenv()

# --- CONFIG ---
URL = input("Enter YouTube URL: ")
MODEL_SIZE = "medium"  # Switched to medium based on your description
SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live JP-to-EN Transcriber & Translator")
        self.root.geometry("1000x700")
        self.root.configure(bg='#121212')

        self.context_memory = []
        self.is_running = True

        self.setup_gui()
        self.run_system_checks()

        # Start AI thread
        self.thread = threading.Thread(target=self.run_ai, daemon=True)
        self.thread.start()

    def setup_gui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, bg='#1e1e1e', pady=10)
        control_frame.pack(fill='x')

        self.status_lbl = tk.Label(control_frame, text="Initializing...", fg='#00ff00', bg='#1e1e1e', font=("Arial", 12, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=15)

        btn_clear_text = tk.Button(control_frame, text="Clear Screen", command=self.clear_screen, bg='#333333', fg='white')
        btn_clear_text.pack(side=tk.RIGHT, padx=5)

        btn_reset_mem = tk.Button(control_frame, text="Reset AI Context", command=self.reset_context, bg='#8b0000', fg='white')
        btn_reset_mem.pack(side=tk.RIGHT, padx=15)

        # Main Text Area
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, bg='#1a1a1a', fg='#e0e0e0', 
            insertbackground='white', font=("Yu Gothic", 14)
        )
        self.text_area.pack(expand=True, fill='both', padx=15, pady=15)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def run_system_checks(self):
        warnings = []
        
        # 1. Hugging Face Token Check
        if not os.path.exists(".env"):
            warnings.append("WARNING: '.env' file is missing! Hugging Face downloads may fail.")
        elif not os.getenv("HF_TOKEN"):
            warnings.append("WARNING: 'HF_TOKEN' is missing inside the '.env' file.")

        # 2. Windows Check
        if platform.system() == "Windows":
            warnings.append("NOTE: Running on Windows. Ensure FFmpeg is installed and added to your system PATH.")

        # 3. GPU & VRAM Check
        if not torch.cuda.is_available():
            warnings.append("CRITICAL: NVIDIA GPU not detected! Torch cannot find CUDA. CPU mode will be very slow.")
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            self.status_lbl.config(text=f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | OS: {platform.system()}")
            
            if vram_gb < 7.5:
                warnings.append(f"WARNING: Detected {vram_gb:.1f}GB VRAM. 8GB+ is recommended for optimal performance.")

        # Show warnings if any
        if warnings:
            warning_text = "\n\n".join(warnings)
            messagebox.showwarning("System Check Warnings", warning_text)

    def clear_screen(self):
        """Clears the GUI text area"""
        self.text_area.delete(1.0, tk.END)

    def reset_context(self):
        """Clears the AI's short-term memory to stop hallucination loops"""
        self.context_memory = []
        self.text_area.insert(tk.END, "\n[SYSTEM: AI Context Memory Reset]\n\n")
        self.text_area.see(tk.END)

    def format_output(self, start_time, jp_text, en_text):
        mins = int(start_time // 60)
        secs = int(start_time % 60)
        timestamp = f"[{mins:02d}:{secs:02d}]"

        kks = pykakasi.kakasi()
        parsed = kks.convert(jp_text)
        
        line_kanji, line_hira = "", ""
        
        for word in parsed:
            orig, hira = word['orig'], word['hira']
            max_len = max(len(orig), len(hira))
            line_kanji += orig.ljust(max_len, '\u3000') + " \u3000"
            line_hira += hira.ljust(max_len, '\u3000') + " \u3000"

        return f"{timestamp}\n{line_kanji}\n{line_hira}\nEN: {en_text}\n{'-'*70}\n"

    def update_gui(self, text_block):
        self.text_area.insert(tk.END, text_block)
        self.text_area.see(tk.END)

    def run_ai(self):
        try:
            self.root.after(0, self.update_gui, f"Loading Whisper ({MODEL_SIZE}) on GPU...\n")
            model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
            
            self.root.after(0, self.update_gui, "Loading Helsinki-NLP Translator on GPU...\n")
            model_name = "Helsinki-NLP/opus-mt-ja-en"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            translator_model = MarianMTModel.from_pretrained(model_name).cuda()

            self.root.after(0, self.update_gui, "Connecting to YouTube stream... Listening...\n======================================================================\n")

            yt_dlp_cmd = ['yt-dlp', '-f', 'bestaudio/best', '--quiet', '--no-warnings', '-o', '-', URL]
            ffmpeg_cmd = [
                'ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ac', '1', 
                '-ar', str(SAMPLE_RATE), '-acodec', 'pcm_s16le', '-loglevel', 'quiet', '-'
            ]
            
            self.yt_process = subprocess.Popen(yt_dlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.ff_process = subprocess.Popen(ffmpeg_cmd, stdin=self.yt_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            while self.is_running:
                raw_audio = self.ff_process.stdout.read(CHUNK_SAMPLES * 2)
                if not raw_audio: break

                audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                segments, _ = model.transcribe(audio_np, language="ja", beam_size=5, vad_filter=True)

                for segment in segments:
                    jp_text = segment.text.strip()
                    if jp_text:
                        self.context_memory.append(jp_text)
                        if len(self.context_memory) > 2:
                            self.context_memory.pop(0)

                        contextual_input = " ".join(self.context_memory)
                        
                        inputs = tokenizer(contextual_input, return_tensors="pt", padding=True).to("cuda")
                        translated_tokens = translator_model.generate(**inputs)
                        translation_result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                        
                        en_sentences = translation_result.split('. ')
                        current_en = en_sentences[-1] if en_sentences else translation_result

                        block = self.format_output(segment.start, jp_text, current_en.strip())
                        self.root.after(0, self.update_gui, block)

        except Exception as e:
            self.root.after(0, self.update_gui, f"\n[ERROR]: {e}\n")
        finally:
            self.on_closing()

    def on_closing(self):
        self.is_running = False
        try:
            self.ff_process.kill()
            self.yt_process.kill()
        except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()