from transformers import pipeline
import torch
import torchaudio
from pydub import AudioSegment
import numpy as np

# Create the pipeline
pipe = pipeline("text-to-speech", model="suno/bark-small", device="cuda:1")

# Your input text
text = "NARRATOR: [clears throat] This is a test ... can anybody hear me?"

# Generate speech
output = pipe(text)

# Extract audio data and sampling rate
audio_data = output["audio"]
sampling_rate = output["sampling_rate"]

# Convert to torch tensor
waveform = torch.tensor(audio_data)

# Save audio file as WAV
torchaudio.save("output.wav", waveform, sampling_rate)
print("Audio file saved as 'output.wav'")

# Convert torch tensor to numpy array
numpy_array = waveform.numpy()

# Ensure the numpy array is in the correct range (-32768 to 32767 for 16-bit audio)
numpy_array = np.int16(numpy_array * 32767)

# Convert numpy array to AudioSegment
audio_segment = AudioSegment(
    numpy_array.tobytes(),
    frame_rate=sampling_rate,
    sample_width=2,  # 2 bytes for 16-bit audio
    channels=1,
)

# Export as MP3
audio_segment.export("output.mp3", format="mp3", bitrate="192k")

print("Audio file saved as 'output.mp3'")
