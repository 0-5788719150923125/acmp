import os

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, BarkModel, pipeline

model_name = "suno/bark-small"
cache_dir = "data"
device = "cuda:1"

processor = AutoProcessor.from_pretrained(
    model_name, cache_dir=cache_dir, device_map=device
)
model = BarkModel.from_pretrained(model_name, cache_dir=cache_dir, device_map=device)

voice_preset = "v2/en_speaker_6"

text = "[clears throat] This is a test ... can you hear me? ... No? ... Okay, good."
inputs = processor(text, voice_preset=voice_preset).to(device)

audio_array = model.generate(**inputs, do_sample=True, temperature=0.3)
sample_rate = model.generation_config.sample_rate


# Extract audio data and sampling rate
# audio_data = output["audio"]
# sampling_rate = output["sampling_rate"]

# Convert to torch tensor
waveform = torch.tensor(audio_array)

# save audio to disk, but first take the sample rate from the model config
audio_array = audio_array.cpu().numpy().squeeze()

# Save audio file as WAV
write_wav("output.wav", sample_rate, audio_array)
print("Audio file saved as 'output.wav'")

# Convert torch tensor to numpy array
numpy_array = waveform.cpu().numpy()

# Ensure the numpy array is in the correct range (-32768 to 32767 for 16-bit audio)
numpy_array = np.int16(numpy_array * 32767)

# Convert numpy array to AudioSegment
audio_segment = AudioSegment(
    numpy_array.tobytes(),
    frame_rate=sample_rate,
    sample_width=2,  # 2 bytes for 16-bit audio
    channels=1,
)

# Export as MP3
audio_segment.export("output.mp3", format="mp3", bitrate="192k")

print("Audio file saved as 'output.mp3'")

# import os

# import numpy as np
# import torch
# import torchaudio
# from pydub import AudioSegment
# from transformers import pipeline

# os.environ["HF_HOME"] = "data"

# # Create the pipeline
# pipe = pipeline("text-to-speech", model="suno/bark-small", device="cuda:1")

# # Your input text
# text = "[clears throat] This is a test ... can you hear me?"

# # Generate speech
# output = pipe(text)

# # Extract audio data and sampling rate
# audio_data = output["audio"]
# sampling_rate = output["sampling_rate"]

# # Convert to torch tensor
# waveform = torch.tensor(audio_data)

# # Save audio file as WAV
# torchaudio.save("output.wav", waveform, sampling_rate)
# print("Audio file saved as 'output.wav'")

# # Convert torch tensor to numpy array
# numpy_array = waveform.numpy()

# # Ensure the numpy array is in the correct range (-32768 to 32767 for 16-bit audio)
# numpy_array = np.int16(numpy_array * 32767)

# # Convert numpy array to AudioSegment
# audio_segment = AudioSegment(
#     numpy_array.tobytes(),
#     frame_rate=sampling_rate,
#     sample_width=2,  # 2 bytes for 16-bit audio
#     channels=1,
# )

# # Export as MP3
# audio_segment.export("output.mp3", format="mp3", bitrate="192k")

# print("Audio file saved as 'output.mp3'")
