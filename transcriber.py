import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import pykakasi
import gc
import unicodedata

class AIEngine:
    def __init__(self, translator_type, source_lang_code, target_lang_code, nllb_source_code, nllb_target_code, helsinki_id=None):
        
        # VRAM Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # --- DEVICE & VRAM CHECKING ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_compute_type = "float16" if self.device == "cuda" else "int8"
        self.translator_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            print(f"[SYSTEM] Detected GPU VRAM: {vram_gb:.2f} GB")
            
            if vram_gb < 3.5:
                print("⚠️ [WARNING] Low VRAM detected. Falling back to CPU.")
                self.device = "cpu"
                self.whisper_compute_type = "int8"
                self.translator_dtype = torch.float32
            else:
                print("✅ [SYSTEM] Sufficient VRAM. Using GPU Acceleration.")
        else:
            print("⚠️ [SYSTEM] No GPU detected. Running on CPU.")

        # Whisper Init (Medium)
        self.whisper_model_size = "medium" 
        self.source_lang = source_lang_code if source_lang_code != "auto" else None
        
        print(f"[AI] Loading Whisper ({self.whisper_model_size}) into {self.device.upper()}...")
        self.whisper = WhisperModel(self.whisper_model_size, device=self.device, compute_type=self.whisper_compute_type)

        # Translator Init
        self.translator_type = translator_type
        self.nllb_target_code = nllb_target_code
        self.nllb_source_code = nllb_source_code
        
        self.kks = pykakasi.kakasi()
        self.context_memory = []
        self.total_processed_seconds = 0.0
        
        # Default Display Options
        self.display_ops = {"kanji": True, "hira": True, "gloss": True, "trans": True}

        if self.translator_type == "helsinki" and helsinki_id:
            print(f"[AI] Loading Helsinki-NLP ({helsinki_id}) into {self.device.upper()} (FP16)...")
            self.tokenizer = MarianTokenizer.from_pretrained(helsinki_id)
            self.translator = MarianMTModel.from_pretrained(
                helsinki_id, 
                torch_dtype=self.translator_dtype
            ).to(self.device)
        else:
            print(f"[AI] Loading Universal NLLB-200 (600M) into {self.device.upper()} (FP16)...")
            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=nllb_source_code)
            self.translator = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                torch_dtype=self.translator_dtype
            ).to(self.device)

    def update_display_options(self, opts):
        """Updates what layers should be shown (Kanji, Hira, Gloss, Sentence)"""
        self.display_ops = opts

    def process_audio(self, audio_chunk):
        chunk_duration = len(audio_chunk) / 16000.0

        segments, info = self.whisper.transcribe(
            audio_chunk, language=self.source_lang, beam_size=5, vad_filter=True
        )

        detected_lang = info.language
        results = []

        for segment in segments:
            text = segment.text.strip()
            if not text: continue

            # --- DOUBLING FIX ---
            input_text = text
            if detected_lang == "ja":
                self.context_memory.append(text)
                if len(self.context_memory) > 2:
                    self.context_memory.pop(0)
                input_text = " ".join(self.context_memory)
            
            # --- FULL SENTENCE TRANSLATION (Only if enabled) ---
            final_trans = ""
            if self.display_ops['trans']:
                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
                
                if self.translator_type == "helsinki":
                    translated_tokens = self.translator.generate(**inputs, max_length=200)
                else:
                    target_id = self.tokenizer.convert_tokens_to_ids(self.nllb_target_code)
                    translated_tokens = self.translator.generate(
                        **inputs, forced_bos_token_id=target_id, max_length=200
                    )
                
                translation_result = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                
                final_trans = translation_result
                if detected_lang == "ja":
                    en_sentences = translation_result.split('. ')
                    final_trans = en_sentences[-1] if en_sentences else translation_result

            # Calculate absolute timestamp
            absolute_start_time = self.total_processed_seconds + segment.start
            
            formatted = self._format_text(absolute_start_time, text, final_trans.strip(), detected_lang)
            results.append(formatted)

        self.total_processed_seconds += chunk_duration

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return results

    def _get_display_width(self, text):
        width = 0
        for char in text:
            if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
                width += 2
            else:
                width += 1
        return width

    def _format_text(self, start_time, source_text, translated_text, detected_lang):
        mins, secs = int(start_time // 60), int(start_time % 60)
        timestamp = f"[{mins:02d}:{secs:02d}]"

        if detected_lang == "ja":
            parsed = self.kks.convert(source_text)
            
            # --- BATCH TRANSLATE WORDS (Only if enabled) ---
            translated_words = []
            if self.display_ops['gloss']:
                words_to_translate = [item['orig'] for item in parsed]
                if words_to_translate:
                    batch_inputs = self.tokenizer(words_to_translate, return_tensors="pt", padding=True).to(self.device)
                    
                    if self.translator_type == "helsinki":
                        batch_generated = self.translator.generate(**batch_inputs, max_length=50)
                    else:
                        target_id = self.tokenizer.convert_tokens_to_ids(self.nllb_target_code)
                        batch_generated = self.translator.generate(
                            **batch_inputs, forced_bos_token_id=target_id, max_length=50
                        )
                    translated_words = self.tokenizer.batch_decode(batch_generated, skip_special_tokens=True)

            line_kanji = ""
            line_hira = ""
            line_gloss = ""
            
            pad_jp = '\u3000' 
            pad_en = ' '      

            for i, word_data in enumerate(parsed):
                orig = word_data['orig']
                hira = word_data['hira']
                gloss = translated_words[i].strip() if i < len(translated_words) else ""

                w_orig = self._get_display_width(orig)
                w_hira = self._get_display_width(hira)
                w_gloss = len(gloss)
                
                # Determine max width based on what is actually shown
                widths = []
                if self.display_ops['kanji']: widths.append(w_orig)
                if self.display_ops['hira']: widths.append(w_hira)
                if self.display_ops['gloss']: widths.append(w_gloss)
                
                max_w = max(widths) + 2 if widths else 2

                if self.display_ops['kanji']:
                    line_kanji += orig + pad_jp * max(1, int((max_w - w_orig)/2)) + " "
                
                if self.display_ops['hira']:
                    line_hira  += hira + pad_jp * max(1, int((max_w - w_hira)/2)) + " "
                
                if self.display_ops['gloss']:
                    line_gloss += gloss.ljust(max_w, pad_en) + "  "

            # Construct Final String based on Toggles
            final_output = f"{timestamp}\n"
            if self.display_ops['kanji']: final_output += f"{line_kanji}\n"
            if self.display_ops['hira']:  final_output += f"{line_hira}\n"
            if self.display_ops['gloss']: final_output += f"{line_gloss}\n"
            if self.display_ops['trans']: final_output += f"TRANS: {translated_text}\n"
            final_output += f"{'-'*70}\n"
            
            return final_output

        else:
            return f"{timestamp}\nSRC: {source_text}\nTRANS: {translated_text}\n{'-'*70}\n"

    def reset_memory(self):
        self.context_memory = []

    def cleanup_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()