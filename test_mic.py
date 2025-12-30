import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

duration = 3  #seconds
fs = 16000  # sample rate

print("🎤 Speak now...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
audio = np.int16(audio * 32767)
write("test_record.wav", fs, audio)
print("✅ Recording saved as test_record.wav")
