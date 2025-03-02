import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import mediapipe as mp
import os
import glob
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable oneDNN custom operations to remove warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    # Convert back to BGR
    preprocessed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return preprocessed

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
    with face_detect.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:  # Lowered confidence threshold
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections is None:
            return img  # Return the original image if no faces are detected
        for i, detection in enumerate(results.detections):
            box = detection.location_data.relative_bounding_box
            x_start, y_start = int(box.xmin * img.shape[1]), int(box.ymin * img.shape[0])
            x_end, y_end = int((box.xmin + box.width) * img.shape[1]), int((box.ymin + box.height) * img.shape[0])
            annotated_img = cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color)
            cv2.putText(annotated_img, text, (x_start - 20, y_start - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        return annotated_img

def conv_to_vid(path, count, outpath):
    img = []
    for i in range(count):
        x = f"{path}/{i}.png"
        img.append(x)

    if not img:
        logging.error("No images found to convert to video.")
        return

    cv2_fourcc = cv2.VideoWriter_fourcc(*'h264')
    frame = cv2.imread(img[0])
    size = list(frame.shape)
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(f"{outpath}/output.mp4", cv2_fourcc, 24, size) 
    for i in range(len(img)): 
        video.write(cv2.imread(img[i]))
    video.release()

def run_ffmpeg(input_path1, input_path2, output_path=r"mixdvid/output.mp4"):
    ffmpeg_command = [
        r"C:\Users\mravi\Documents\ffmpeg-7.1-essentials_build[1]\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe", 
        "-i", input_path1, 
        "-vf", f"movie={input_path2}, scale=250:-1 [inner]; [in][inner] overlay=10:10 [out]", 
        output_path
    ]
    try:
        subprocess.run(ffmpeg_command, check=True)
        logging.info(f"FFmpeg command executed successfully. Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running FFmpeg: {e}")

def removefilesinfold(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)

def trim_video(input_path, output_path, duration=7):
    with VideoFileClip(input_path) as video:
        logging.info(f"Original video FPS: {video.fps}")
        trimmed_video = video.subclip(0, min(duration, video.duration))
        # Ensure that audio is correctly encoded by specifying an audio_codec (e.g., "aac")
        trimmed_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=video.fps or 24)

def run(path):
    logging.info(f"Processing (original) video: {path}")
    
    vid = cv2.VideoCapture(path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Original total frames: {total_frames}")

    resconfarr = []
    count = 0
    result = []
    frame_number = 0
    prev_frame = None

    # Process only first 10 seconds (assuming 24 fps → 240 frames)
    max_frames = 240

    while vid.isOpened() and frame_number < max_frames:
        suc, input_image = vid.read()
        if not suc:
            logging.error(f"Failed to read frame {frame_number}. Exiting.")
            break

        logging.info(f"Processing frame {frame_number}, shape: {input_image.shape}")
        if prev_frame is not None and np.array_equal(input_image, prev_frame):
            logging.warning(f"Frame {frame_number} is identical to previous frame.")
        prev_frame = input_image.copy()

        # Process frame (preprocess, face detection, etc.)
        input_image = preprocess_frame(input_image)
        face = mtcnn(input_image)
        if face is None:
            logging.warning(f"No face detected on frame {frame_number}.")
            frame_number += 1
            continue

        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
        face = face.to(DEVICE).to(torch.float32) / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        target_layers = [model.block8.branch1[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            real_prediction = output.item()
            confidences = {'real': real_prediction, 'fake': 1 - real_prediction}
            logging.info(f"Frame {frame_number} - real: {real_prediction}")

        cv2.imwrite(f"tempinppics/{count}.png", input_image)
        cv2.imwrite(f"boxpics/{count}.png", box(input_image, real_prediction))
        cv2.imwrite(f"pics/{count}.png", face_with_mask)
        result.append(1 if real_prediction < 0.6 else 0)
        frame_number += 1
        count += 1

    vid.release()  # Ensure the video capture object is released

    # Generate output video from processed images
    conv_to_vid("pics", count, "vid")
    conv_to_vid("boxpics", count, "boxvid")
    run_ffmpeg(r"boxvid/output.mp4", r"vid/output.mp4")
    outpath = "mixdvid/output.mp4"
    
    # Optionally check that the file was created
    if not os.path.exists(outpath):
        logging.error(f"Output video file not found: {outpath}")

    if len(result) == 0:
        final_res = [0, 0]
    else:
        final_res = [sum(result) / len(result), (len(result) - sum(result)) / len(result)]
    return resconfarr, outpath, final_res