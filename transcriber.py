import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import pykakasi
import gc

class AIEngine:
    def __init__(self, translator_type, source_lang_code, target_lang_code, nllb_source_code, nllb_target_code, helsinki_id=None):
        
        # VRAM Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # --- DEVICE & VRAM CHECKING ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_compute_type = "float16" if self.device == "cuda" else "int8"
        # Only use float16 for translators if we are on CUDA to save massive amounts of VRAM
        self.translator_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            # total_memory returns bytes. Convert to GB.
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            print(f"[SYSTEM] Detected GPU VRAM: {vram_gb:.2f} GB")
            
            # Whisper Med + NLLB takes ~3GB. If VRAM is too low, we fallback to CPU.
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
        self.context_memory =[]

        if self.translator_type == "helsinki" and helsinki_id:
            print(f"[AI] Loading Helsinki-NLP ({helsinki_id}) into {self.device.upper()} (FP16)...")
            self.tokenizer = MarianTokenizer.from_pretrained(helsinki_id)
            # Load in float16 to cut VRAM usage in half
            self.translator = MarianMTModel.from_pretrained(
                helsinki_id, 
                torch_dtype=self.translator_dtype
            ).to(self.device)
        else:
            print(f"[AI] Loading Universal NLLB-200 (600M) into {self.device.upper()} (FP16)...")
            model_name = "facebook/nllb-200-distilled-600M"
            # NLLB requires src_lang at initialization
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=nllb_source_code)
            # Load in float16 to cut VRAM usage in half (Drops from 2.5GB to ~1.2GB)
            self.translator = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                torch_dtype=self.translator_dtype
            ).to(self.device)

    def process_audio(self, audio_chunk):
        segments, info = self.whisper.transcribe(
            audio_chunk, language=self.source_lang, beam_size=5, vad_filter=True
        )

        detected_lang = info.language
        results =[]

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
            
            # Ensure inputs are moved to the correct dynamically checked device
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
            
            # Generate Translation
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

            formatted = self._format_text(segment.start, text, final_trans.strip(), detected_lang)
            results.append(formatted)

        # --- CRITICAL FIX: FREE VRAM FOR WHISPER ---
        # PyTorch hogs memory after translation. We must force it to release 
        # the unused memory so CTranslate2 (Whisper) doesn't OOM on the next loop.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return results

    def _format_text(self, start_time, source_text, translated_text, detected_lang):
        mins, secs = int(start_time // 60), int(start_time % 60)
        timestamp = f"[{mins:02d}:{secs:02d}]"

        if detected_lang == "ja":
            parsed = self.kks.convert(source_text)
            line_kanji, line_hira = "", ""
            for word in parsed:
                orig, hira = word['orig'], word['hira']
                max_len = max(len(orig), len(hira))
                line_kanji += orig.ljust(max_len, '\u3000') + " \u3000"
                line_hira += hira.ljust(max_len, '\u3000') + " \u3000"
            return f"{timestamp}\n{line_kanji}\n{line_hira}\nTRANS: {translated_text}\n{'-'*70}\n"
        else:
            return f"{timestamp}\nSRC: {source_text}\nTRANS: {translated_text}\n{'-'*70}\n"

    def reset_memory(self):
        self.context_memory =