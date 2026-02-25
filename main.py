import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from dotenv import load_dotenv
import queue
import json
import os
import gc
import torch

from audio_capture import AudioCapture
from transcriber import AIEngine
import model_manager

load_dotenv()

LANGUAGES = [
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
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)

class NewInputWindow:
    def __init__(self, parent, main_gui):
        self.top = tk.Toplevel(parent)
        self.top.title("New Input Configuration")
        self.top.geometry("750x900")
        self.top.configure(bg='#121212')
        self.main_gui = main_gui
        self.history = load_history()
        
        # UI Variables
        self.src_var = tk.StringVar(value="1") 
        self.url_var = tk.StringVar()
        self.live_var = tk.StringVar(value="static")
        self.engine_var = tk.StringVar(value="1")
        self.src_lang_var = tk.StringVar(value="3") # Default JP for demo
        self.tgt_lang_var = tk.StringVar(value="1") # Default EN
        
        # Display Options (Default All True)
        self.show_kanji_var = tk.BooleanVar(value=True)
        self.show_hira_var = tk.BooleanVar(value=True)
        self.show_gloss_var = tk.BooleanVar(value=True)
        self.show_trans_var = tk.BooleanVar(value=True)

        self.build_ui()
        self.update_ui()
        
        # Bind traces
        self.src_var.trace_add("write", lambda *args: self.update_ui())
        self.engine_var.trace_add("write", lambda *args: self.update_ui())

    def create_toggle_btn(self, parent, text, var, value, color, width=15):
        rb = tk.Radiobutton(
            parent, text=text, variable=var, value=value,
            indicatoron=False, bg='#1e1e1e', fg='white', 
            selectcolor=color, activebackground=color, activeforeground='white',
            font=("Arial", 10, "bold"), width=width, pady=5, cursor="hand2"
        )
        return rb

    def create_list_btn(self, parent, prefix, label, var, value, color, font=("Arial", 10)):
        full_text = f"[{prefix}]  {label}"
        rb = tk.Radiobutton(
            parent, text=full_text, variable=var, value=value,
            indicatoron=False, bg='#1e1e1e', fg='white', 
            selectcolor=color, activebackground=color, activeforeground='white',
            font=font, width=60, anchor='w', padx=15, pady=4, cursor="hand2"
        )
        return rb

    def create_check_btn(self, parent, text, var, color):
        cb = tk.Checkbutton(
            parent, text=text, variable=var,
            bg='#121212', fg='white', selectcolor='#1e1e1e',
            activebackground='#121212', activeforeground='white',
            font=("Arial", 10), pady=5
        )
        return cb

    def build_ui(self):
        self.container = tk.Frame(self.top, bg='#121212')
        self.container.pack(fill='both', expand=True, padx=20, pady=10)

        # --- 1. SOURCE ---
        self.frame_1_source = tk.Frame(self.container, bg='#121212')
        self.frame_1_source.pack(fill='x', pady=(10,0))
        tk.Label(self.frame_1_source, text="1. Source :", bg='#121212', fg='#aaaaaa', font=("Arial", 11, "bold")).pack(anchor='w', pady=(0,5))
        r_frame = tk.Frame(self.frame_1_source, bg='#121212')
        r_frame.pack(anchor='w')
        self.create_toggle_btn(r_frame, "YouTube", self.src_var, "1", "#c62828").pack(side='left', padx=(0,10))
        self.create_toggle_btn(r_frame, "Local File", self.src_var, "2", "#f57c00").pack(side='left', padx=(0,10))
        self.create_toggle_btn(r_frame, "Microphone", self.src_var, "3", "#2e7d32").pack(side='left', padx=(0,10))

        # --- 2.1 URL ---
        self.frame_2_url = tk.Frame(self.container, bg='#121212')
        tk.Label(self.frame_2_url, text="2.1 Input Source (URL or Path) :", bg='#121212', fg='#aaaaaa', font=("Arial", 10, "bold")).pack(anchor='w')
        tk.Entry(self.frame_2_url, textvariable=self.url_var, bg='#1e1e1e', fg='white', insertbackground='white', font=("Arial", 12)).pack(fill='x', pady=5)

        # --- 2.1.1 LIVE ---
        self.frame_3_live = tk.Frame(self.container, bg='#121212')
        tk.Label(self.frame_3_live, text="2.1.1 Static video or a livestream? :", bg='#121212', fg='#aaaaaa', font=("Arial", 10, "bold")).pack(anchor='w', pady=(0,5))
        l_frame = tk.Frame(self.frame_3_live, bg='#121212')
        l_frame.pack(anchor='w')
        self.create_toggle_btn(l_frame, "Static Video", self.live_var, "static", "#0277bd").pack(side='left', padx=(0,10))
        self.create_toggle_btn(l_frame, "Livestream", self.live_var, "live", "#c62828").pack(side='left', padx=(0,10))

        # --- 3. MODEL ---
        self.frame_4_engine = tk.Frame(self.container, bg='#121212')
        self.frame_4_engine.pack(fill='x', pady=(20,0))
        tk.Label(self.frame_4_engine, text="3. Model type? :", bg='#121212', fg='#aaaaaa', font=("Arial", 11, "bold")).pack(anchor='w', pady=(0, 5))
        
        self.create_list_btn(self.frame_4_engine, "1", "Helsinki-NLP (Specific pairs, faster)", self.engine_var, "1", "#1b5e20", font=("Consolas", 11)).pack(anchor='w', pady=2)
        self.create_list_btn(self.frame_4_engine, "2", "NLLB-200 (Universal, fall-back)", self.engine_var, "2", "#1b5e20", font=("Consolas", 11)).pack(anchor='w', pady=2)
        
        tk.Frame(self.frame_4_engine, height=1, bg='#333333').pack(fill='x', pady=8)
        
        for i, entry in enumerate(self.history):
            idx_str = str(i + 3)
            label = f"[HISTORY] {entry['type'].upper()} | {entry['src_name']} -> {entry['tgt_name']}"
            self.create_list_btn(self.frame_4_engine, idx_str, label, self.engine_var, idx_str, "#006064", font=("Consolas", 10)).pack(anchor='w', pady=1)

        # --- 3.1 LANG ---
        self.frame_5_lang = tk.Frame(self.container, bg='#121212')
        col1 = tk.Frame(self.frame_5_lang, bg='#121212')
        col1.pack(side='left', expand=True, fill='both', padx=(0, 10))
        tk.Label(col1, text="(List of language source)", bg='#121212', fg='#aaaaaa', font=("Arial", 10, "bold")).pack(anchor='w', pady=(0,5))
        for i, lang in enumerate(LANGUAGES):
            self.create_list_btn(col1, str(i+1), lang['name'], self.src_lang_var, str(i+1), "#0d47a1", font=("Arial", 10)).pack(anchor='w', pady=1)
            
        col2 = tk.Frame(self.frame_5_lang, bg='#121212')
        col2.pack(side='left', expand=True, fill='both')
        tk.Label(col2, text="(List of language output)", bg='#121212', fg='#aaaaaa', font=("Arial", 10, "bold")).pack(anchor='w', pady=(0,5))
        for i, lang in enumerate(LANGUAGES):
            self.create_list_btn(col2, str(i+1), lang['name'], self.tgt_lang_var, str(i+1), "#4a148c", font=("Arial", 10)).pack(anchor='w', pady=1)

        # --- 4. DISPLAY OPTIONS (For Japanese/Generic) ---
        self.frame_6_disp = tk.Frame(self.container, bg='#121212')
        self.frame_6_disp.pack(fill='x', pady=(20,0))
        tk.Label(self.frame_6_disp, text="4. Output Layers (Japanese Only) :", bg='#121212', fg='#aaaaaa', font=("Arial", 11, "bold")).pack(anchor='w', pady=(0,5))
        
        d_frame = tk.Frame(self.frame_6_disp, bg='#121212')
        d_frame.pack(anchor='w')
        self.create_check_btn(d_frame, "Kanji (Original)", self.show_kanji_var, "#1e1e1e").pack(side='left', padx=(0,15))
        self.create_check_btn(d_frame, "Hiragana (Reading)", self.show_hira_var, "#1e1e1e").pack(side='left', padx=(0,15))
        self.create_check_btn(d_frame, "Literal Word Meaning", self.show_gloss_var, "#1e1e1e").pack(side='left', padx=(0,15))
        self.create_check_btn(d_frame, "Sentence Translation", self.show_trans_var, "#1e1e1e").pack(side='left', padx=(0,15))


        # --- RUN BUTTON ---
        btn_frame = tk.Frame(self.top, bg='#121212')
        btn_frame.pack(fill='x', side='bottom')
        tk.Button(btn_frame, text="RUN ->", bg='#2962ff', fg='white', font=('Arial', 14, 'bold'), width=12, pady=5, cursor="hand2", command=self.run).pack(side='right', padx=20, pady=20)

    def update_ui(self):
        self.frame_2_url.pack_forget()
        self.frame_3_live.pack_forget()
        self.frame_5_lang.pack_forget()

        s = self.src_var.get()
        e = self.engine_var.get()

        if s in ["1", "2"]:
            self.frame_2_url.pack(fill='x', pady=(15,0), after=self.frame_1_source)
        if s == "1":
            self.frame_3_live.pack(fill='x', pady=(15,0), after=self.frame_2_url)
            
        if e in ["1", "2"]:
            self.frame_5_lang.pack(fill='x', pady=(20,0), after=self.frame_4_engine)

    def run(self):
        s_choice = self.src_var.get()
        s_data = self.url_var.get().strip()
        is_live = (self.live_var.get() == "live")
        e_val = int(self.engine_var.get())

        if s_choice in ["1", "2"] and not s_data:
            messagebox.showerror("Error", "Please enter an Input Source (URL or File Path).")
            return
        
        if e_val >= 3:
            idx = e_val - 3
            if idx >= len(self.history): return
            sel = self.history[idx]
            t_type = sel['type']
            s_code = sel['src_code']
            t_code = sel['tgt_code']
            nllb_src = sel.get('nllb_src')
            nllb_tgt = sel['nllb_tgt']
            h_id = sel.get('helsinki_id')
        else:
            s_idx = int(self.src_lang_var.get()) - 1
            t_idx = int(self.tgt_lang_var.get()) - 1
            src_lang = LANGUAGES[s_idx]
            tgt_lang = LANGUAGES[t_idx]

            t_type = "helsinki" if e_val == 1 else "nllb"
            h_id = None

            if t_type == "helsinki":
                matches = model_manager.find_helsinki_models(src_lang['name'], tgt_lang['name'])
                if matches: h_id = matches[0]['id'] 
                else: t_type = "nllb"
            
            s_code = src_lang['code']
            t_code = tgt_lang['code']
            nllb_src = src_lang['nllb']
            nllb_tgt = tgt_lang['nllb']

            new_entry = {
                "type": t_type, "src_name": src_lang["name"], "src_code": s_code,
                "nllb_src": nllb_src, "tgt_name": tgt_lang["name"], "tgt_code": t_code,
                "nllb_tgt": nllb_tgt, "helsinki_id": h_id
            }
            self.history = [h for h in self.history if not (h['src_name'] == new_entry['src_name'] and h['tgt_name'] == new_entry['tgt_name'] and h['type'] == new_entry['type'])]
            self.history.insert(0, new_entry)
            save_history(self.history[:7])

        # Get Display Options
        disp_opts = {
            "kanji": self.show_kanji_var.get(),
            "hira": self.show_hira_var.get(),
            "gloss": self.show_gloss_var.get(),
            "trans": self.show_trans_var.get()
        }

        self.main_gui.apply_new_settings(s_choice, s_data, is_live, t_type, s_code, t_code, nllb_src, nllb_tgt, h_id, disp_opts)
        self.top.destroy()


class MainGUI:
    def __init__(self, root, audio_cap):
        self.root = root
        self.root.title("Transcriber GUI")
        self.root.geometry("900x650")
        self.root.configure(bg='#121212')
        self.audio_cap = audio_cap
        self.ai_engine = None # Initialized after settings are chosen
        
        # UI Setup
        control_frame = tk.Frame(self.root, bg='#1e1e1e', pady=10)
        control_frame.pack(fill='x')
        
        tk.Button(control_frame, text="New Input / Change Settings", command=self.open_new_input_window, bg='#2e7d32', fg='white').pack(side=tk.LEFT, padx=15)
        
        tk.Button(control_frame, text="Clear Screen", command=self.clear_screen).pack(side=tk.RIGHT, padx=5)
        # STOP BUTTON replaces Reset Memory
        tk.Button(control_frame, text="STOP", command=self.stop_capture, bg='#c62828', fg='white', width=10).pack(side=tk.RIGHT, padx=15)

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, bg='#1a1a1a', fg='#e0e0e0', font=("Yu Gothic", 14))
        self.text_area.pack(expand=True, fill='both', padx=15, pady=15)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.is_running = True
        self.thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.thread.start()

    def open_new_input_window(self):
        NewInputWindow(self.root, self)

    def stop_capture(self):
        """Stops the audio capture but keeps AI loaded"""
        if self.audio_cap.is_capturing:
            self.audio_cap.stop()
            self.update_gui("\n[SYSTEM] Capture Stopped by User.\n")

    def apply_new_settings(self, s_choice, s_data, is_live, t_type, s_code, t_code, nllb_src, nllb_tgt, h_id, disp_opts):
        # 1. Stop Audio
        self.audio_cap.stop()
        while not self.audio_cap.audio_queue.empty():
            try: self.audio_cap.audio_queue.get_nowait()
            except queue.Empty: break
        
        # 2. Force Reload AI
        self.update_gui(f"\n[SYSTEM] Reloading AI Models...\n{'='*50}\n")
        
        if self.ai_engine:
            del self.ai_engine
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        self.ai_engine = AIEngine(t_type, s_code, t_code, nllb_src, nllb_tgt, h_id)
        # Apply the selected display options
        self.ai_engine.update_display_options(disp_opts)
        
        self.update_gui("[SYSTEM] AI Model Loaded. Starting Audio...\n")

        # 3. Start Audio
        self.src_choice = s_choice
        self.src_data = s_data
        self.is_live = is_live
        self._finished_notified = False
        
        self.start_audio()

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
        # self.update_gui("Waiting for input configuration...\n")
        while self.is_running:
            try:
                # If AI isn't loaded yet (start of app), just wait
                if not self.ai_engine:
                    # time.sleep(0.5) 
                    # We can't import time here without adding import, 
                    # but Queue get with timeout handles the wait naturally
                    pass 
                
                audio_chunk = self.audio_cap.audio_queue.get(timeout=1.0)
                
                if self.ai_engine:
                    results = self.ai_engine.process_audio(audio_chunk)
                    for res in results:
                        self.update_gui(res)
            except queue.Empty: 
                if getattr(self.audio_cap, 'is_capturing', False) is False and self.ai_engine:
                    if getattr(self, '_finished_notified', False) is False:
                        self.update_gui("\n[SYSTEM] Media playback finished or stopped. Cache freed.\n")
                        self.ai_engine.cleanup_cache()
                        self._finished_notified = True
                continue
            except Exception as e:
                pass # safely ignore setup timing errors

    def update_gui(self, text):
        self.root.after(0, lambda: self._insert_text(text))

    def _insert_text(self, text):
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)

    def clear_screen(self): self.text_area.delete(1.0, tk.END)
    def on_closing(self):
        self.is_running = False
        self.audio_cap.stop()
        self.root.destroy()

if __name__ == "__main__":
    # No more CLI prompts here
    audio_module = AudioCapture()
    
    root = tk.Tk()
    app = MainGUI(root, audio_module)
    
    # Launch the settings window immediately on startup
    root.after(100, app.open_new_input_window)
    root.mainloop()