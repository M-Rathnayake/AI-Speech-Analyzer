#speech to text
import whisper
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        """Initialize Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium)
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text or None if error
        """
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

# Example usage
if __name__ == "__main__":
    transcriber = AudioTranscriber("base")
    text = transcriber.transcribe("audio_samples/sample.mp3")
    print(text if text else "Transcription failed")