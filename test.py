# import mediapipe as mp
# import cv2
# def box(img,conf):
#     text=""
    
#     if conf>=0.5:
#         text="real"
#         color=(0,255,0)
#     else:
#         text="false"
#         color=(255,0,0)
#     face_detect=mp.solutions.face_detection
#     mp_drawing=mp.solutions.drawing_utils
#     with face_detect.FaceDetection(
#         model_selection=1,min_detection_confidence=0.5) as face_detection:
#         results=face_detection.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#         for i,detection in enumerate(results.detections):
#             box=detection.location_data.relative_bounding_box
#             x_start,y_start=int(box.xmin*img.shape[1]),int(box.ymin*img.shape[0])
#             x_end,y_end=int((box.xmin+box.width)*img.shape[1]),int((box.ymin+box.height)*img.shape[0])
#             annotated_img=cv2.rectangle(img,(x_start,y_start),(x_end,y_end),color)
#             cv2.putText(annotated_img,text,(x_start-20,y_start-20),cv2.FONT_HERSHEY_DUPLEX,1,color,2)
#         return annotated_img

#TODO

import subprocess

# def run_ffmpeg(input_path1, input_path2, output_path=r"output.mp4"):
#     """
#     Executes an FFmpeg command to overlay a video onto another video.

#     Parameters:
#     - input_path1: str, path to the main input video file.
#     - input_path2: str, path to the overlay video file.
#     - output_path: str, path for the output video file (default is 'vid\output.mp4').
#     """
#     # Define your FFmpeg command with the input paths
#     ffmpeg_command = [
#         r"c:\ffmpeg\ffmpeg-master-latest-win64-gpl\ffmpeg.exe", 
#         "-i", input_path1,  # Main input video
#         "-vf", f"movie={input_path2}, scale=250:-1 [inner]; [in][inner] overlay=10:10 [out]", 
#         output_path
#     ]

#     try:
#         # Execute the FFmpeg command
#         subprocess.run(ffmpeg_command, check=True)
#         print(f"FFmpeg command executed successfully. Output saved to: {output_path}")
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred while running FFmpeg: {e}")

# # Example usage
# main_video_path = r"boxvid/output.mp4"    # Replace with your main video path, use raw string format
# overlay_video_path = r"boxvid/output.mp4"   # Replace with your overlay video path, use raw string format

# # Run the function
# run_ffmpeg(main_video_path, overlay_video_path)
#TODO
# def has_audio(filename):
#     result = subprocess.run([r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\ffprobe.exe", "-v", "error", "-show_entries",
#                              "format=nb_streams", "-of",
#                              "default=noprint_wrappers=1:nokey=1", filename],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT)
#     return (int(result.stdout) -1)
# import streamlit as st
# import pandas as pd
# import numpy as np
# print(pd.DataFrame(np.random.randn(1,2)))
#TODO
# import streamlit as st
# from audio_recorder_streamlit import audio_recorder
# import os
# from io import BytesIO
# from pydub import AudioSegment

# import gradio as gr
# import torch
# import torch.nn.functional as F
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import numpy as np
# from PIL import Image
# import cv2
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import warnings
# import os
# import glob
# import mediapipe as mp
# import subprocess
# import streamlit as st
# import io
# from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
# import torch
# import librosa
# import torch.nn.functional as F
# from moviepy.editor import VideoFileClip
# st.write("audio")
# model_name = "MelodyMachine/Deepfake-audio-detection-V2"
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
# model1 = AutoModelForAudioClassification.from_pretrained(model_name)


# audio_bytes = audio_recorder()

# def classify_recording(path):
#     # Load the feature extractor and model


#     # Load and preprocess the audio file
#     audio, sr = librosa.load(BytesIO(path), sr=16000)  # Ensure the correct sample rate

#     # Process the audio file to extract features
#     inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

#     # Run inference
#     with torch.no_grad():
#         outputs = model1(**inputs)

#     # Extract and print the predicted label
#     predicted_label = torch.argmax(outputs.logits, dim=-1)





#     confidences = F.softmax(outputs.logits, dim=1)
#     percentages = confidences * 100
#     print(percentages.tolist())
#     return percentages.tolist()
# import streamlit as st
# import soundcard as sc
# import soundfile as sf
# run= st.button("run")
# if run:
#     OUTPUT_FILE_NAME = "out.wav"    # file name.
#     SAMPLE_RATE = 48000              # [Hz]. sampling rate.
#     RECORD_SEC = 10                  # [sec]. duration recording audio.

#     with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
#         # record audio with loopback from default speaker.
#         data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
        
#         # change "data=data[:, 0]" to "data=data", if you would like to write audio as multiple-channels.
#         sf.write(file=OUTPUT_FILE_NAME, data=data[:, 0], samplerate=SAMPLE_RATE)
# if audio_bytes:
   
#     st.audio(audio_bytes) 
#     x=classify_recording(audio_bytes)
#     st.write(f"{x}")


import torch
print(torch.cuda.is_available())