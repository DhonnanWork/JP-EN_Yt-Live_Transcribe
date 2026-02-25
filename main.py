import threading
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from dotenv import load_dotenv
import queue
import json
import os
import requests

from audio_capture import AudioCapture
from transcriber import AIEngine
import model_manager

load_dotenv()

LANGUAGES =[
    {"name": "English", "code": "en", "nllb": "eng_Latn"},
    {"name": "Indonesian", "code": "id", "nllb": "ind_Latn"},
    {"name": "Japanese", "code": "ja", "nllb": "jpn_Jpan"},
    {"name": "Spanish", "code": "es", "nllb": "spa_Latn"},
    {"name": "French", "code": "fr", "nllb": "fra_Latn"},
    {"name": "German", "code": "de", "nllb": "deu_Latn"},
    {"name": "Chinese", "code": "zh", "nllb": "zho_Hans"},
    {"name": "Korean", "code": "ko", "nllb": "kor_Hang"},
]

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f: return json.load(f)
        except: return []
    return[]

def save_history(history):
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

def setup_terminal_prompts():
    print("\n" + "="*50)
    print(" LIVE TRANSCRIBER & TRANSLATOR (RTX 5050 Optimized)")
    print("="*50)
    
    # --- STEP 1: INPUT ---
    print("\n[STEP 1] Select Input Source:")
    print("1. YouTube")
    print("2. Local Audio/Video File")
    print("3. Microphone")
    source_choice = input("Enter 1, 2, or 3: ").strip()

    source_data = ""
    is_live = False

    if source_choice == "1":
        source_data = input("Enter YouTube URL: ").strip()
        live_input = input("Is this a LIVE Stream? (y/n): ").strip().lower()
        is_live = (live_input == 'y')
    elif source_choice == "2":
        source_data = input("Enter File Path: ").strip()

    # --- STEP 2: HISTORY ---
    history = load_history()
    print("\n[STEP 2] Select Translation Engine:")
    print("1. Helsinki-NLP (Specific pairs, faster)")
    print("2. NLLB-200 (Universal, fall-back)")
    
    if history:
        print("-" * 30)
        for i, entry in enumerate(history):
            print(f"{i+3}. [HISTORY] {entry['type'].upper()} | {entry['src_name']} -> {entry['tgt_name']}")
        print("-" * 30)

    choice = input(f"Enter 1-{2 + len(history)}: ").strip()

    if choice.isdigit() and int(choice) >= 3:
        idx = int(choice) - 3
        if idx < len(history):
            sel = history[idx]
            print(f"\nLoaded from history: {sel['type'].upper()} ({sel['src_name']} -> {sel['tgt_name']})")
            return source_choice, source_data, is_live, sel['type'], sel['src_code'], sel['tgt_code'], sel.get('nllb_src'), sel['nllb_tgt'], sel.get('helsinki_id')

    translator_type = "helsinki" if choice == "1" else "nllb"
    helsinki_id = None

    print("\n--- Select Source Language ---")
    for i, lang in enumerate(LANGUAGES): print(f"{i+1}. {lang['name']}")
    src_idx = int(input("Enter number: ").strip()) - 1
    src_lang = LANGUAGES[src_idx]

    print("\n--- Select Target Language ---")
    for i, lang in enumerate(LANGUAGES): print(f"{i+1}. {lang['name']}")
    tgt_idx = int(input("Enter number: ").strip()) - 1
    tgt_lang = LANGUAGES[tgt_idx]

    if translator_type == "helsinki":
        print(f"\nChecking database for {src_lang['name']} -> {tgt_lang['name']}...")
        matches = model_manager.find_helsinki_models(src_lang['name'], tgt_lang['name'])
        
        if matches:
            print("\n✅ Found Helsinki Models:")
            print("-" * 50)
            for i, m in enumerate(matches):
                print(f"{i+1}. {m['id']}  ({m['src']} -> {m['tgt']})")
            print("-" * 50)
            m_choice = input(f"Choose model (1-{len(matches)}): ").strip()
            helsinki_id = matches[int(m_choice)-1]['id']
        else:
            print(f"\n⚠️  No direct Helsinki model found.")
            print("🔄 Switching to NLLB-200 automatically.")
            translator_type = "nllb"

    new_entry = {
        "type": translator_type,
        "src_name": src_lang["name"],
        "src_code": src_lang["code"],
        "nllb_src": src_lang["nllb"],
        "tgt_name": tgt_lang["name"],
        "tgt_code": tgt_lang["code"],
        "nllb_tgt": tgt_lang["nllb"],
        "helsinki_id": helsinki_id
    }
    
    history = [h for h in history if not (h['src_name'] == new_entry['src_name'] and h['tgt_name'] == new_entry['tgt_name'] and h['type'] == new_entry['type'])]
    history.insert(0, new_entry)
    save_history(history[:7])

    return source_choice, source_data, is_live, translator_type, src_lang['code'], tgt_lang['code'], src_lang['nllb'], tgt_lang['nllb'], helsinki_id

class MainGUI:
    def __init__(self, root, audio_cap, ai_engine, src_choice, src_data, is_live):
        self.root = root
        self.root.title("Transcriber GUI")
        self.root.geometry("900x600")
        self.root.configure(bg='#121212')
        self.audio_cap = audio_cap
        self.ai_engine = ai_engine
        
        self.src_choice = src_choice
        self.src_data = src_data
        self.is_live = is_live

        control_frame = tk.Frame(self.root, bg='#1e1e1e', pady=10)
        control_frame.pack(fill='x')
        
        # --- NEW INPUT BUTTON ---
        tk.Button(control_frame, text="New Input", command=self.ask_new_input, bg='#2e7d32', fg='white').pack(side=tk.LEFT, padx=15)
        
        tk.Button(control_frame, text="Clear Screen", command=self.clear_screen).pack(side=tk.RIGHT, padx=5)
        tk.Button(control_frame, text="Reset AI Memory", command=self.reset_mem, bg='darkred', fg='white').pack(side=tk.RIGHT, padx=15)

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, bg='#1a1a1a', fg='#e0e0e0', font=("Yu Gothic", 14))
        self.text_area.pack(expand=True, fill='both', padx=15, pady=15)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.is_running = True
        self.thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.thread.start()

        self.root.after(100, self.start_audio)

    def ask_new_input(self):
        # 1. Stop whatever is currently playing
        self.audio_cap.stop()
        
        # 2. Clear the audio queue so old audio doesn't bleed over
        while not self.audio_cap.audio_queue.empty():
            try: self.audio_cap.audio_queue.get_nowait()
            except queue.Empty: break
            
        # 3. Force free GPU Memory
        self.ai_engine.cleanup_cache()

        # 4. Ask the user for a new URL/File/Mic
        new_source = simpledialog.askstring(
            "New Input", 
            "Enter a YouTube URL, Local File Path, or type 'mic':"
        )
        
        if not new_source:
            self.update_gui("\n[SYSTEM] Input cancelled. Standing by...\n")
            return

        new_source = new_source.strip()
        self._finished_notified = False
        self.update_gui(f"\n\n[SYSTEM] Loading new source: {new_source}\n{'='*50}\n")
        
        # Route to the correct module
        if new_source.lower() == 'mic':
            self.src_choice = "3"
            self.audio_cap.start_mic()
        elif new_source.startswith("http") or "youtube.com" in new_source or "youtu.be" in new_source:
            self.src_choice = "1"
            self.src_data = new_source
            is_live_str = simpledialog.askstring("Live Stream", "Is this a LIVE stream? (y/n)")
            self.is_live = True if is_live_str and is_live_str.lower() == 'y' else False
            self.audio_cap.start_youtube(self.src_data, self.is_live, status_callback=self.download_progress_callback)
        else:
            self.src_choice = "2"
            self.src_data = new_source
            self.audio_cap.start_file(self.src_data)

    def download_progress_callback(self, text):
        clean_text = text.strip()
        if clean_text:
            self.root.after(0, lambda: self._update_download_status(clean_text))

    def _update_download_status(self, text):
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)

    def start_audio(self):
        if self.src_choice == "1":
            self.audio_cap.start_youtube(self.src_data, self.is_live, status_callback=self.download_progress_callback)
        elif self.src_choice == "2":
            self.audio_cap.start_file(self.src_data)
        elif self.src_choice == "3":
            self.audio_cap.start_mic()

    def processing_loop(self):
        self._finished_notified = False
        self.update_gui("Listening for audio...\n" + "="*50 + "\n")
        while self.is_running:
            try:
                audio_chunk = self.audio_cap.audio_queue.get(timeout=1.0)
                results = self.ai_engine.process_audio(audio_chunk)
                for res in results:
                    self.update_gui(res)
            except queue.Empty: 
                # Check if the audio file finished natively (queue is empty & capture stopped)
                if not getattr(self.audio_cap, 'is_capturing', False):
                    if getattr(self, '_finished_notified', False) is False:
                        self.update_gui("\n[SYSTEM] Media playback finished. GPU memory cache freed.\n")
                        self.ai_engine.cleanup_cache()
                        self._finished_notified = True
                continue

    def update_gui(self, text):
        self.root.after(0, lambda: self._insert_text(text))

    def _insert_text(self, text):
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)

    def clear_screen(self): self.text_area.delete(1.0, tk.END)
    def reset_mem(self): self.ai_engine.reset_memory(); self.update_gui("\n[SYSTEM: AI Context Memory Reset]\n\n")
    def on_closing(self):
        self.is_running = False
        self.audio_cap.stop()
        self.root.destroy()

if __name__ == "__main__":
    s_choice, s_data, is_live, t_type, s_code, t_code, nllb_src, nllb_tgt, h_id = setup_terminal_prompts()

    print("\nBooting Modules... (Please wait while AI models load into VRAM)")
    audio_module = AudioCapture()
    ai_module = AIEngine(t_type, s_code, t_code, nllb_src, nllb_tgt, h_id)

    print("Launching GUI...")
    root = tk.Tk()
    app = MainGUI(root, audio_module, ai_module, s_choice, s_data, is_live)
    root.mainloop()