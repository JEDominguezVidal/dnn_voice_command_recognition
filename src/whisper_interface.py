import whisper
import torch
import numpy as np

class WhisperInterface:
    """
    Interface for OpenAI Whisper speech recognition model.
    
    Handles model loading, transcription, and GPU optimization.
    
    Args:
        model_size (str): Whisper model size (e.g., 'tiny', 'small')
        device (str): Preferred computation device ('cuda' or 'cpu')
        fallback_size (str): Model size to use if OOM occurs
    """
    def __init__(self, model_size="small", device="cuda", fallback_size="tiny"):
        self.model_size = model_size
        self.fallback_size = fallback_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        
    def _load_model(self):
        """
        Load Whisper model with GPU optimizations.
        
        Returns:
            whisper.Whisper: Loaded Whisper model instance
            
        Raises:
            RuntimeError: If model loading fails and no fallback available
        """
        try:
            model = whisper.load_model(self.model_size, device=self.device)
            
            # Apply GPU optimizations
            if self.device == "cuda":
                model = model.half()  # FP16 for faster inference
                
            return model
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.model_size != self.fallback_size:
                print(f"Falling back to {self.fallback_size} model due to OOM")
                return whisper.load_model(self.fallback_size, device=self.device)
            raise
    
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio (np.ndarray): Audio data as int16 numpy array
            sr (int): Sample rate (default 16000)
            
        Returns:
            str: Recognized text transcript in lowercase
        """
        # Convert audio to float32 and normalize
        audio = audio.astype(np.float32) / 32768.0
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio,
            language='en',
            fp16=(self.device == "cuda"),
            temperature=0.0  # Deterministic output
        )
        return result["text"].strip().lower()

def map_to_command(transcript: str, command_list: list) -> str:
    """
    Map Whisper transcript to the best matching predefined command.
    
    Args:
        transcript (str): Text transcript from Whisper
        command_list (list): List of valid command strings
        
    Returns:
        str: The matched command or 'unknown' if no match found
    """
    transcript = transcript.lower()
    
    # First pass: exact substring match
    for cmd in command_list:
        if cmd in transcript:
            return cmd
            
    # Second pass: handle special cases with synonyms
    # (Commented out since it wasn't working reliably)
    # if "stop" in command_list and any(w in transcript for w in ["halt", "pause"]):
    #     return "stop"
    # if "go" in command_list and "proceed" in transcript:
    #     return "go"
    # if "yes" in command_list and "affirmative" in transcript:
    #     return "yes"
    # if "no" in command_list and "negative" in transcript:
    #     return "no"
            
    return "unknown"
