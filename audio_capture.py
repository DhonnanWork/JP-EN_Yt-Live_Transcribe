import subprocess
import numpy as np
import sounddevice as sd
import queue
import threading
import os
import time

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_seconds=8):
        self.sample_rate = sample_rate
        self.chunk_samples = sample_rate * chunk_seconds
        self.audio_queue = queue.Queue()
        self.is_capturing = False
        self.process = None
        self.temp_filename = "temp_vod.wav"

    def get_live_stream_url(self, url):
        try:
            # -g gets the URL.
            # -f ba/b ensures that if an audio-only stream (ba) is missing (common in livestreams), 
            # it falls back to the best combined stream (b). FFmpeg will just strip the video anyway.
            cmd = ['yt-dlp', '-g', '-f', 'ba/b', url]
            
            # Anti-Bot Headers
            cmd.extend(['--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'])
            
            # --- YT-DLP COOKIE INJECTION (BYPASS BLOCKING) - COMMENTED OUT FOR NOW ---
            # cookie_path = os.getenv("YTDLP_COOKIES")
            # if cookie_path and os.path.exists(cookie_path):
            #     cmd.extend(['--cookies', cookie_path])
            
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                return res.stdout.strip()
            else:
                print(f"[YT-DLP Error]: {res.stderr}")
        except Exception as e:
            print(f"[Error extracting Live URL]: {e}")
        return None

    def _download_vod(self, url, status_callback=None):
        if os.path.exists(self.temp_filename):
            try: os.remove(self.temp_filename)
            except: pass
            
        if status_callback:
            status_callback("[System] Initializing Download...\n")

        cmd =[
            # Also updated fallback to ba/b here to prevent VODs from failing
            'yt-dlp', '-f', 'ba/b', 
            '-x', '--audio-format', 'wav', 
            '-o', 'temp_vod.%(ext)s',
            '--postprocessor-args', f'ffmpeg:-ar {self.sample_rate} -ac 1',
            '--force-overwrites',
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            '--no-check-certificate'
        ]

        # --- YT-DLP COOKIE INJECTION (BYPASS BLOCKING) - COMMENTED OUT FOR NOW ---
        # cookie_path = os.getenv("YTDLP_COOKIES")
        # if cookie_path and os.path.exists(cookie_path):
        #     cmd.extend(['--cookies', cookie_path])

        # Always append URL *last* when passing arguments to subprocess
        cmd.append(url)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            if status_callback:
                if "[download]" in line: status_callback(line.strip())
                elif "ERROR" in line: status_callback(f"[YT-DLP LOG]: {line.strip()}")

        process.wait()

        if os.path.exists("temp_vod.wav"): return "temp_vod.wav"
        return None

    def _process_ffmpeg_stream(self, input_source):
        ffmpeg_cmd =[
            'ffmpeg', '-i', input_source, 
            '-f', 's16le', '-ac', '1',
            '-ar', str(self.sample_rate), '-acodec', 'pcm_s16le', '-loglevel', 'quiet', '-'
        ]
        
        self.process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        while self.is_capturing:
            raw_audio = self.process.stdout.read(self.chunk_samples * 2)
            if not raw_audio: break
            
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_np)
        
        if input_source == "temp_vod.wav" and os.path.exists(input_source):
            try: os.remove(input_source)
            except: pass
        self.stop()

    def _mic_callback(self, indata, frames, time, status):
        if self.is_capturing: self.audio_queue.put(indata.copy().flatten())

    def start_youtube(self, url, is_live, status_callback=None):
        self.is_capturing = True
        if is_live:
            if status_callback: status_callback("[Audio] Live Mode: Connecting to stream...\n")
            stream_url = self.get_live_stream_url(url)
            if stream_url:
                threading.Thread(target=self._process_ffmpeg_stream, args=(stream_url,), daemon=True).start()
            else:
                if status_callback: status_callback("⚠️ Failed to get Live URL.\n")
                self.is_capturing = False
        else:
            threading.Thread(target=self._handle_vod_download_and_play, args=(url, status_callback), daemon=True).start()

    def _handle_vod_download_and_play(self, url, status_callback):
        filename = self._download_vod(url, status_callback)
        if filename:
            if status_callback: status_callback(f"\n[Audio] Download complete. Processing...\n{'='*50}\n")
            self._process_ffmpeg_stream(filename)
        else:
            if status_callback: status_callback("\n[Critical Error] Download failed.\n")
            self.is_capturing = False

    def start_file(self, file_path):
        self.is_capturing = True
        threading.Thread(target=self._process_ffmpeg_stream, args=(file_path,), daemon=True).start()

    def start_mic(self):
        self.is_capturing = True
        self.mic_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', blocksize=self.chunk_samples, callback=self._mic_callback)
        self.mic_stream.start()

    def stop(self):
        self.is_capturing = False
        if self.process: self.process.kill()
        if hasattr(self, 'mic_stream'): self.mic_stream.stop(); self.mic_stream.close()
        if os.path.exists(self.temp_filename):
            try: os.remove(self.temp_filename)
            except: pass