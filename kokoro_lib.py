import kokoro
from kokoro import KPipeline
import soundfile as sf
import torch
import threading
import queue
import time
from typing import Optional, Generator, Tuple
import numpy as np
import sounddevice as sd

class KokoroTTSAgent:
    def __init__(self, lang_code: str = 'a', voice: str = 'af_heart', speed: float = 1.2):
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.pipeline = KPipeline(lang_code=self.lang_code)
        self.is_initialized = False
        self.sample_rate = 24000
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            print(f"🔄 Chargement du modèle Kokoro (lang: {self.lang_code}, voice: {self.voice})...")
            start_time = time.time()
            load_time = time.time() - start_time
            print(f"✅ Modèle Kokoro chargé avec succès en {load_time:.2f}s")
            self.is_initialized = True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            self.is_initialized = False

    def _play_audio(self, audio):
        """Plays audio safely using OutputStream to avoid CFFI callback errors."""
        # Handle both Tensor and numpy array
        if hasattr(audio, 'numpy'):
            audio_float = audio.numpy().astype(np.float32)
        else:
            audio_float = audio.astype(np.float32)
        
        # Ensure 1D
        if audio_float.ndim > 1:
            audio_float = audio_float.squeeze()

        finished = threading.Event()

        def callback(outdata, frames, time_info, status):
            nonlocal audio_float
            chunk = audio_float[:frames]
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk.reshape(-1, 1)
                outdata[len(chunk):] = 0
                finished.set()
                raise sd.CallbackStop()
            else:
                outdata[:] = chunk.reshape(-1, 1)
                audio_float = audio_float[frames:]

        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=callback,
            finished_callback=finished.set
        ):
            finished.wait()

    def _collect_and_play(self, generator):
        start_time = time.time()
        audio_segments = []

        for i, (gs, ps, audio) in enumerate(generator):
            # Convert tensor to numpy if needed
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            audio_segments.append(audio)
            if i == 0:
                print(f"⚡ Premier segment généré en {time.time() - start_time:.3f}s")

        if not audio_segments:
            print("⚠️ Aucun segment audio généré")
            return np.array([])

        full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]

        total_time = time.time() - start_time
        audio_duration = len(full_audio) / self.sample_rate
        rtf = total_time / audio_duration
        print(f"🎵 Audio généré: {audio_duration:.2f}s en {total_time:.3f}s (RTF: {rtf:.3f})")

        print("🔊 Lecture en direct (audio unifié)...")
        self._play_audio(full_audio)

        generator.close()
        return full_audio

    def synthesize_text(self, text: str, save_path: Optional[str] = None, play_direct: bool = True) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Le modèle n'est pas initialisé correctement")

        try:
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)

            if play_direct:
                full_audio = self._collect_and_play(generator)
            else:
                audio_segments = []
                for i, (gs, ps, audio) in enumerate(generator):
                    # In both places where you iterate the generator, add:
                    if hasattr(audio, 'numpy'):
                        audio = audio.numpy()
                    audio_segments.append(audio)

                full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
                generator.close()

            if save_path:
                sf.write(save_path, full_audio, self.sample_rate)
                print(f"💾 Audio sauvegardé: {save_path}")

            return full_audio

        except Exception as e:
            print(f"❌ Erreur lors de la synthèse: {e}")
            raise

    def synthesize_streaming(self, text: str, play_direct: bool = True) -> Generator[Tuple[str, str, np.ndarray], None, None]:
        if not self.is_initialized:
            raise RuntimeError("Le modèle n'est pas initialisé correctement")

        try:
            start_time = time.time()
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)

            segments = []
            for i, (gs, ps, audio) in enumerate(generator):
                # In both places where you iterate the generator, add:
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                
                if i == 0:
                    print(f"⚡ Premier segment en streaming: {time.time() - start_time:.3f}s")
                segments.append((gs, ps, audio))
                yield gs, ps, audio

            if play_direct and segments:
                full_audio = np.concatenate([a for _, _, a in segments])
                print("🔊 Lecture unifiée après streaming...")
                self._play_audio(full_audio)

            generator.close()

        except Exception as e:
            print(f"❌ Erreur lors du streaming: {e}")
            raise

    def start_background_processing(self):
        if self.is_processing:
            print("⚠️ Le traitement en arrière-plan est déjà actif")
            return
        self.is_processing = True
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
        print("🚀 Traitement en arrière-plan démarré")

    def _background_worker(self):
        while self.is_processing:
            try:
                text_item = self.text_queue.get(timeout=1.0)
                if text_item is None:
                    break
                text, callback_id, play_direct = text_item
                audio = self.synthesize_text(text, play_direct=play_direct)
                self.audio_queue.put((callback_id, audio))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erreur dans le worker: {e}")

    def queue_text(self, text: str, callback_id: str = None, play_direct: bool = True) -> str:
        if callback_id is None:
            callback_id = f"tts_{int(time.time() * 1000)}"
        self.text_queue.put((text, callback_id, play_direct))
        return callback_id

    def get_audio_result(self, timeout: float = 5.0) -> Optional[Tuple[str, np.ndarray]]:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_background_processing(self):
        if not self.is_processing:
            return
        self.is_processing = False
        self.text_queue.put(None)
        if hasattr(self, 'background_thread'):
            self.background_thread.join(timeout=2.0)
        print("🛑 Traitement en arrière-plan arrêté")

    def wait_for_audio_completion(self):
        """Waits until current audio playback finishes (no-op with OutputStream approach)."""
        pass

    def change_voice(self, voice: str):
        self.voice = voice
        print(f"🎤 Voix changée: {voice}")

    def change_speed(self, speed: float):
        self.speed = speed
        print(f"⚡ Vitesse changée: {speed}")

    def get_status(self) -> dict:
        return {
            'initialized': self.is_initialized,
            'lang_code': self.lang_code,
            'voice': self.voice,
            'speed': self.speed,
            'background_processing': self.is_processing,
            'queue_size': self.text_queue.qsize() if hasattr(self, 'text_queue') else 0,
        }

    def __del__(self):
        self.stop_background_processing()