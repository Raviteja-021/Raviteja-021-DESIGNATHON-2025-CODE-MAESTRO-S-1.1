import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import subprocess
import streamlit as st
import io
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
import soundfile as sf

st.set_page_config(page_title='deepfake classification', layout='wide', initial_sidebar_state="collapsed")

# Load deepfake detection model
model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model1 = AutoModelForAudioClassification.from_pretrained(model_name)

def onchange():
    removefilesinfold("tempvid")
    removefilesinfold("temppics")
    removefilesinfold("tempaudio")
    removefilesinfold("output_of_photo")

def removefilesinfold(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)

def save_uploaded_file(uploaded_file):
    # Save the file temporarily
    save_path = os.path.join("tempaudio", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def preprocess_audio(file_path):
    try:
        # Load audio with librosa (for common formats)
        audio_array, sr = librosa.load(file_path, sr=16000, mono=True)
        return audio_array
    except Exception as e:
        st.warning(f"Librosa failed: {e}. Trying soundfile...")

    try:
        # Try using soundfile (for other formats)
        audio_array, sr = sf.read(file_path)
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        return audio_array
    except Exception as e:
        st.warning(f"Soundfile failed: {e}. Trying FFmpeg...")

    # Convert to WAV using FFmpeg as a last resort
    output_file = "converted.wav"
    try:
        subprocess.run(["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", output_file, "-y"], check=True)
        audio_array, sr = librosa.load(output_file, sr=16000, mono=True)
        return audio_array
    except Exception as e:
        st.error(f"FFmpeg conversion failed: {e}")
        return None  # Return None if all methods fail

def classify_audio(path):
    audio_array = preprocess_audio(path)

    if audio_array is not None:
        # Ensure it's a NumPy array with the correct shape
        audio_array = np.array(audio_array).astype(np.float32)

        # Trim or pad the audio to 1 second (16000 samples)
        if len(audio_array) > 16000:
            audio_input = audio_array[:16000]  # Trim
        else:
            audio_input = np.pad(audio_array, (0, max(0, 16000 - len(audio_array))))  # Pad

        # Run prediction
        try:
            inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = model1(**inputs)
            confidences = F.softmax(outputs.logits, dim=1)
            percentages = confidences * 100
            return percentages.tolist()
        except Exception as pipe_err:
            st.error(f"Pipeline error: {pipe_err}")
            return None
    else:
        st.error("❌ Could not process the audio file.")
        return None

def classify_recording(audio_bytes):
    audio_array, sr = librosa.load(BytesIO(audio_bytes), sr=16000)

    # Ensure it's a NumPy array with the correct shape
    audio_array = np.array(audio_array).astype(np.float32)

    # Trim or pad the audio to 1 second (16000 samples)
    if len(audio_array) > 16000:
        audio_input = audio_array[:16000]  # Trim
    else:
        audio_input = np.pad(audio_array, (0, max(0, 16000 - len(audio_array))))  # Pad

    # Run prediction
    try:
        inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model1(**inputs)
        confidences = F.softmax(outputs.logits, dim=1)
        percentages = confidences * 100
        return percentages.tolist()
    except Exception as pipe_err:
        st.error(f"Pipeline error: {pipe_err}")
        return None

st.write(" # $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ Audio:studio_microphone:")
with st.container(border=True):
    uploaded_file = st.file_uploader("Choose an audio file...", on_change=onchange)

col1, col2 = st.columns(2)
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    with col1:
        with st.container(border=True):
            st.audio(file_path)
            classify = st.button("Classify")
    if classify:
        audconf = classify_audio(file_path)
        if audconf:
            with col2:
                with st.container(border=True):
                    st.write(f"## Real percentage={float('{:.3f}'.format(audconf[0][0]))}")
                    st.progress(audconf[0][0] / 100)
                with st.container(border=True):
                    st.write(f"## Fake percentage={float('{:.3f}'.format(audconf[0][1]))}")
                    st.progress(audconf[0][1] / 100)

audio_bytes = audio_recorder(icon_size="1.5x")
if audio_bytes:
    with st.container(border=True):
        with col1:
            st.audio(audio_bytes)
            classify = st.button("Classify", key=2)
    if classify:
        x = classify_recording(audio_bytes)
        if x:
            with col2:
                with st.container(border=True):
                    st.write(f"# **Real percentage**={float('{:.2f}'.format(x[0][1]))}")
                    st.progress(x[0][1] / 100)
                with st.container(border=True):
                    st.write(f"# **Fake percentage**={float('{:.2f}'.format(x[0][0]))}")
                    st.progress(x[0][0] / 100)




