import whisper
from typing import Optional, Callable
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """Initialize Whisper model with enhanced configuration.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium)
            device: Compute device ('auto', 'cpu', or 'cuda')
        """
        self.device = self._determine_device(device)
        self.model = self._load_model(model_size)
        self.progress_callback = None

    def _determine_device(self, device: str) -> str:
        """Automatically select the best available device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, model_size: str):
        """Load model with optimized settings."""
        logger.info(f"Loading {model_size} model for {self.device.upper()}")
        model = whisper.load_model(model_size, device=self.device)
        
        # Optimize for CPU if needed
        if self.device == "cpu":
            model = model.float()  # Force FP32
            logger.info("Optimized for CPU execution (FP32)")
        
        return model

    def set_progress_callback(self, callback: Callable[[float], None]):
        """Set a progress callback function (0.0 to 1.0)."""
        self.progress_callback = callback

    def _update_progress(self, progress: float):
        """Handle progress updates safely."""
        if self.progress_callback:
            try:
                self.progress_callback(min(1.0, max(0.0, progress)))
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Optional[str]:
        """Enhanced audio transcription with progress tracking.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (en, fr, etc.)
            
        Returns:
            Transcribed text or None if error
        """
        try:
            self._update_progress(0.1)  # Initialization
            
            # Load audio with progress
            self._update_progress(0.2)
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Mel spectrogram
            self._update_progress(0.3)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language if not specified
            self._update_progress(0.4)
            if language is None:
                _, probs = self.model.detect_language(mel)
                language = max(probs, key=probs.get)
                logger.info(f"Detected language: {language}")
            
            # Decode with simulated progress
            self._update_progress(0.5)
            options = whisper.DecodingOptions(language=language, fp16=(self.device != "cpu"))
            
            with tqdm(total=100, disable=(self.progress_callback is not None)) as pbar:
                result = self.model.transcribe(audio_path, decoding_options=options)
                pbar.update(100)
            
            self._update_progress(0.9)
            return result["text"]
            
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            self._update_progress(1.0)
            return None
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            self._update_progress(1.0)
            raise  # Re-raise for upstream handling
        finally:
            self._update_progress(1.0)  # Ensure completion

# Example usage with progress tracking
if __name__ == "__main__":
    def print_progress(progress: float):
        print(f"\rProgress: {progress:.0%}", end="", flush=True)
    
    transcriber = AudioTranscriber("base")
    transcriber.set_progress_callback(print_progress)
    
    try:
        text = transcriber.transcribe("audio_samples/sample.mp3")
        print(f"\nResult: {text}" if text else "\nTranscription failed")
    except Exception as e:
        print(f"\nError: {e}")