import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
import os
import glob
import mediapipe as mp
import subprocess
import streamlit as st
import io
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
import pandas as pd
warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
st.set_page_config(page_title='deepfake classification', layout='wide',initial_sidebar_state="collapsed")
    
# Load model directly
# from transformers import AutoModelForImageClassification,pipeline

# model = AutoModelForImageClassification.from_pretrained("not-lain/deepfake", trust_remote_code=True)

# pipe = pipeline(model="deepfake",trust_remote_code=True,device=DEVICE)

def onchange():
    removefilesinfold("tempvid")
    removefilesinfold("temppics")
    removefilesinfold("tempaudio")
    removefilesinfold("output_of_photo")


mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()
model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model1 = AutoModelForAudioClassification.from_pretrained(model_name)


def box(img, conf):
    text = ""
    
    if conf >= 0.7:
        text = "real"
        color = (0, 255, 0)
    else:
        text = "fake"
        color = (255, 0, 0)
    
    face_detect = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with face_detect.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            for i, detection in enumerate(results.detections):
                box = detection.location_data.relative_bounding_box
                x_start, y_start = int(box.xmin * img.shape[1]), int(box.ymin * img.shape[0])
                x_end, y_end = int((box.xmin + box.width) * img.shape[1]), int((box.ymin + box.height) * img.shape[0])
                annotated_img = cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color)
                cv2.putText(annotated_img, text, (x_start - 20, y_start - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            return annotated_img
        else:
            return img
def conv_to_vid(path, count, outpath):
    img = []

    for i in range(count):
        x = f"{path}/{i}.png"
        img.append(x)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'h264')

    frame = cv2.imread(img[0])
    if frame is None:
        raise FileNotFoundError(f"Image file {img[0]} not found or could not be read.")
    
    size = list(frame.shape)
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(f"{outpath}/output.mp4", cv2_fourcc, 24, size)
    for i in range(len(img)):
        frame = cv2.imread(img[i])
        if frame is None:
            raise FileNotFoundError(f"Image file {img[i]} not found or could not be read.")
        video.write(frame)

    video.release()

def run_ffmpeg(input_path1, input_path2, output_path=r"mixdvid/output.mp4"):
    # Define your FFmpeg command with the input paths
    ffmpeg_command = [
        r"C:\Users\mravi\Downloads\ffmpeg\ffmpeg-2025-02-26-git-99e2af4e78-essentials_build\bin\ffmpeg.exe", 
        "-i", input_path1,  # Main input video
        "-vf", f"movie={input_path2}, scale=250:-1 [inner]; [in][inner] overlay=10:10 [out]", 
        output_path
    ]

    try:
        # Execute the FFmpeg command
        subprocess.run(ffmpeg_command, check=True)
        print(f"FFmpeg command executed successfully. Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running FFmpeg: {e}")

main_video_path = r"boxvid/output.mp4"    
overlay_video_path = r"boxvid/output.mp4"  


def extract_audio(path):    
    mp4_file = path
    mp3_file = "tempaudio/audio.mp3"
    video_clip = VideoFileClip(mp4_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(mp3_file)
    audio_clip.close()
    video_clip.close()
    return mp3_file
 
def removefilesinfold(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)

def run(path):
    removefilesinfold('mixdvid')
    vid = cv2.VideoCapture(path)
    resconfarr=[]
    count=0
    result=[]
    while vid.isOpened():
        curres=[]
        suc,input_image= vid.read()
        if suc==False:
            vid.release()
            continue

        face = mtcnn(input_image)
        if face is None:
            continue
        face = face.unsqueeze(0) # add the batch dimension
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        
        # convert the face into a numpy array to be able to plot it
        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        prev_face = prev_face.astype('uint8')

        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers=[model.block8.branch1[-1]]
        use_cuda = True if torch.cuda.is_available() else False
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = "real" if output.item() < 0.5 else "fake"
            real_prediction = output.item()
            fake_prediction = 1 - output.item()
            
            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }
        y,preimg=confidences, face_with_mask
        
        cv2.imwrite(f"tempinppics/{count}.png",input_image)
        real_prediction =y["real"]
        preimg=face_with_mask
        curres.append(y["real"])
        curres.append(y["fake"])

        resconfarr.append(curres)
        if y["real"]>0.5:
            result.append(1)
        else :
            result.append(0)
        boximg=box(input_image,real_prediction)
        cv2.imwrite(f"boxpics/{count}.png",boximg)
        cv2.imwrite(f"pics/{count}.png",preimg)
        count+=1
    conv_to_vid("pics",count,"vid")
    conv_to_vid("boxpics",count,"boxvid")
    run_ffmpeg(r"boxvid/output.mp4",r"vid/output.mp4")
   
    outpath="mixdvid/output.mp4"
    real=0
    fake=0
    for i in result:
        if i==1:
            real+=1
        else:
            fake+=1.1
    
    real=real/len(result)
    fake=fake/len(result)
    
    show_res=[]
    show_res.append(real)
    show_res.append(fake)

    removefilesinfold('boxpics')
    removefilesinfold("tempinppics")
    
    removefilesinfold('pics')

    return resconfarr,outpath,show_res

def save_uploaded_file(uploaded_file):
    
    save_path = os.path.join("tempvid", uploaded_file.name)  
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def classify_audio(path):
    audio, sr = librosa.load(path, sr=16000)  
    # Process the audio file to extract features
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model1(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=-1)

    confidences = F.softmax(outputs.logits, dim=1)
    percentages = confidences * 100
    print(percentages.tolist())
    return percentages.tolist()


def preprocess_audio(file_path):
    try:
        # Load audio with librosa (for common formats)
        audio_array, sr = librosa.load(file_path, sr=16000, mono=True)
        return audio_array
    except Exception as e:
        print(f"Librosa failed: {e}. Trying soundfile...")

    try:
        # Try using soundfile (for other formats)
        audio_array, sr = sf.read(file_path)
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        return audio_array
    except Exception as e:
       print(f"Soundfile failed: {e}. Trying FFmpeg...")

    # Convert to WAV using FFmpeg as a last resort
    output_file = "converted.wav"
    try:
        subprocess.run(["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", output_file, "-y"], check=True)
        audio_array, sr = librosa.load(output_file, sr=16000, mono=True)
        return audio_array
    except Exception as e:
        st.error(f"FFmpeg conversion failed: {e}")
        return None  # Return None if all methods fail

def has_audio(filename):
    result = subprocess.run([r"C:\Users\mravi\Downloads\ffmpeg\ffmpeg-2025-02-26-git-99e2af4e78-essentials_build\bin\ffprobe.exe", "-v", "error", "-show_entries",
                             "format=nb_streams", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return (int(result.stdout) -1)
#video    

st.write(" # $~$ Video:video_camera:")


with st.container(border= True):
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"],on_change=onchange)
col1,col2=st.columns(2)
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    # with col1:
    #     with st.container(border= True):
    #         st.video(file_path)
    #         classify=st.button("classify")
    if(has_audio(file_path)):
        audio=extract_audio(file_path)
        with col1:
            with st.container(border= True):
                st.video(file_path)
                
                st.audio(audio)
                aud_on_or_of=st.toggle("classify audio")
                classify=st.button("classify")
        
        if(aud_on_or_of and classify):
            confid,outvideo,showconf=run(file_path)
            audconf=classify_audio(audio)
            removefilesinfold('tempvid')
            with col2:
                with st.container(border= True):
                    st.video(outvideo)
                with st.container(border= True):
                    st.write(f"# Real: {float("{:.3f}".format(showconf[0]*100))}%")
                    st.progress(showconf[0]/100)
                    st.write(f"# Fake: {float("{:.3f}".format(showconf[1]*100))}%")
                    st.progress(showconf[1]/100)
                with st.container(border= True):
                    chart_data = pd.DataFrame(confid, columns=["real", "fake"])

                    st.line_chart(chart_data)

            st.write(f"## audio results:")
            with st.container(border= True):
                st.write(f"# Real percentage={float("{:.3f}".format(audconf[0][1]))}")
                st.progress(audconf[0][1]/100)
                st.write(f"# Fake percentage={float("{:.3f}".format(audconf[0][0]))}%")
                st.progress(audconf[0][0]/100)
            
        elif(classify):
            confid,outvideo,showconf=run(file_path)
            removefilesinfold('tempvid')
            with col2:
                with st.container(border= True):
                    st.video(outvideo)
                with st.container(border= True):
                    st.write(f"# Real: {float("{:.3f}".format(showconf[0]))}%")
                    st.write(f"# Fake: {float("{:.3f}".format(showconf[1]))}%")
                with st.container(border= True):
                    chart_data = pd.DataFrame(confid, columns=["real", "fake"])

                    st.line_chart(chart_data)
    else:
        with col1:
            with st.container(border= True):
                st.video(file_path)
                classify=st.button("classify")
        if(classify):
            confid,outvideo,showconf=run(file_path)
            removefilesinfold('tempvid')
            with col2:
                with st.container(border= True):
                    st.video(outvideo)
                with st.container(border= True):
                    st.write(f"## Real: {float("{:.3f}".format(showconf[0]))}%")
                    st.write(f"## Fake: {float("{:.3f}".format(showconf[1]))}%")
                with st.container(border= True):    
                    chart_data = pd.DataFrame(confid, columns=["real", "fake"])

                    st.line_chart(chart_data,y=['fake'])