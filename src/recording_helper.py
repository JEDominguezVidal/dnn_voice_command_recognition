import pyaudio
import numpy as np

# Global constants
FRAMES_PER_BUFFER = 800
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

def record_audio():
    """
    Record audio from the default microphone for a fixed duration.
    
    Returns:
        np.ndarray: Recorded audio as int16 numpy array
    """
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)


def record_chunk_audio(frames, time, frames_per_buffer, rate):
    """
    Record an audio chunk and maintain a rolling buffer.
    
    Args:
        frames (list): Current audio frames buffer
        time (float): Duration in seconds for each chunk
        frames_per_buffer (int): Number of frames per buffer
        rate (int): Audio sample rate
        
    Returns:
        list: Updated frames buffer with new chunk
    """
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer
    )
    frames_edited = frames

    number_chunks = int(rate / frames_per_buffer * time)
    if (len(frames_edited)) < number_chunks:
        data = stream.read(frames_per_buffer)
        frames_edited.append(data)
    else:
        frames_edited.pop(0)
        data = stream.read(frames_per_buffer)
        frames_edited.append(data)

    stream.stop_stream()
    stream.close()

    return frames_edited

def convert_frames_to_audio(frames):
    """
    Convert list of audio frames to a single numpy array.
    
    Args:
        frames (list): List of audio byte frames
        
    Returns:
        np.ndarray: Combined audio as int16 numpy array
    """
    return np.frombuffer(b''.join(frames), dtype=np.int16)


def terminate():
    """
    Terminate the PyAudio session. Call when done with recording.
    """
    p.terminate()
