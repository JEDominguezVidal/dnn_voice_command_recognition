import pyaudio
import numpy as np

FRAMES_PER_BUFFER = 800
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    #print("start recording...")

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
        print(len(frames))

    # print("recording stopped")

    stream.stop_stream()
    stream.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)


def record_chunk_audio(frames, time, frames_per_buffer, rate):
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
        print("WARNING: audio buffer not completed yet")
    else:
        frames_edited.pop(0)
        data = stream.read(frames_per_buffer)
        frames_edited.append(data)


    stream.stop_stream()
    stream.close()

    return frames_edited

def convert_frames_to_audio(frames):
    return np.frombuffer(b''.join(frames), dtype=np.int16)


def terminate():
    p.terminate()
