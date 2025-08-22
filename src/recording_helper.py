import pyaudio
import numpy as np

class AudioRecorder:
    """
    Handles continuous audio recording with persistent stream.
    
    Args:
        rate (int): Sample rate (default 16000)
        frames_per_buffer (int): Chunk size for audio reads (default 800)
    """
    def __init__(self, rate=16000, frames_per_buffer=800):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = rate
        self.FRAMES_PER_BUFFER = frames_per_buffer
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.FRAMES_PER_BUFFER,
            start=False
        )
    
    def start(self):
        """Start the audio stream."""
        self.stream.start_stream()
    
    def read_samples(self, num_samples):
        """
        Read exactly num_samples from the audio stream.
        
        Args:
            num_samples (int): Number of samples to read
            
        Returns:
            np.ndarray: Recorded audio as int16 numpy array
        """
        frames = []
        remaining = num_samples
        
        while remaining > 0:
            to_read = min(self.FRAMES_PER_BUFFER, remaining)
            data = self.stream.read(to_read)
            frames.append(data)
            remaining -= to_read
            
        return np.frombuffer(b''.join(frames), dtype=np.int16)
    
    def close(self):
        """Close the audio stream and terminate resources."""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
