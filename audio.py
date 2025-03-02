import torch
import librosa
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import subprocess
from moviepy.editor import VideoFileClip
import numpy as np
import soundfile as sf
import os

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model1 = AutoModelForAudioClassification.from_pretrained(model_name)

def extract_audio(path):
    mp4_file = path
    wav_file = "tempaudio/audio.wav"  # Use WAV file for lossless conversion
    video_clip = VideoFileClip(mp4_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(wav_file, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()
    return wav_file

def preprocess_audio(file_path):
    try:
        # Try loading with librosa
        audio_array, sr = librosa.load(file_path, sr=16000, mono=True)
        return audio_array
    except Exception as e:
        print(f"Librosa failed: {e}. Trying soundfile...")
    
    try:
        # Try using soundfile as fallback
        audio_array, sr = sf.read(file_path)
        audio_array = np.asarray(audio_array)
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        return audio_array
    except Exception as e:
        print(f"Soundfile failed: {e}. Trying FFmpeg...")
        
    # Last resort: use FFmpeg to convert to WAV
    output_file = "converted.wav"
    try:
        subprocess.run(["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", output_file, "-y"], check=True)
        audio_array, sr = librosa.load(output_file, sr=16000, mono=True)
        return audio_array
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        return None

def classify_audio(path):
    # Preprocess and get the audio array
    audio_array = preprocess_audio(path)
    if audio_array is not None:
        # Ensure it's a NumPy array with type float32
        audio_array = np.array(audio_array).astype(np.float32)
        # Trim or pad to exactly 1 second (16,000 samples)
        if len(audio_array) > 16000:
            audio_input = audio_array[:16000]
        else:
            audio_input = np.pad(audio_array, (0, 16000 - len(audio_array)))
        try:
            inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = model1(**inputs)
            confidences = F.softmax(outputs.logits, dim=1)
            percentages = confidences * 100
            return percentages.tolist()
        except Exception as pipe_err:
            print(f"Pipeline error: {pipe_err}")
            return None
    else:
        print("Could not process the audio file.")
        return None

def has_audio(filename):
    result = subprocess.run([r"C:\Users\mravi\Downloads\ffmpeg\ffmpeg-2025-02-26-git-99e2af4e78-essentials_build\bin\ffprobe.exe",
                             "-v", "error", "-show_entries", "format=nb_streams",
                             "-of", "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    try:
        num_streams = int(result.stdout.decode().strip())
        return (num_streams - 1)
    except Exception as e:
        print(f"Error checking audio streams: {e}")
        return 0