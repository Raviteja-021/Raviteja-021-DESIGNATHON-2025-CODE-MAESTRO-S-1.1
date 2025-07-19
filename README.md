Of course. Here is a comprehensive README file tailored specifically for your project. It includes a project overview, key features, detailed setup instructions, and usage guidelines, structured to be clear and professional.

-----

# Deepfake Detection System

[](https://www.python.org/downloads/)
[](https://streamlit.io)
[](https://opensource.org/licenses/MIT)

A comprehensive deepfake detection tool that analyzes video and audio streams to identify manipulated media. This application leverages multiple deep learning models to provide a robust classification of whether a video is real or a deepfake.

The user-friendly interface is built with Streamlit, allowing users to simply upload a video and receive a detailed analysis, including frame-by-frame confidence scores and visual explanations.

-----

## Key Features

  * **Dual-Mode Analysis**: Simultaneously analyzes both the video frames and the audio track for a more reliable prediction.
  * **🎥 Video Deepfake Detection**:
      * Utilizes an **InceptionResnetV1** model fine-tuned for face classification.
      * Employs **MTCNN** for fast and accurate face detection in each frame.
      * Generates **Grad-CAM** visualizations to show which facial regions the model focused on for its decision.
      * Overlays a bounding box (red for fake, green for real) and the Grad-CAM heatmap onto the output video.
  * **🔊 Audio Deepfake Detection**:
      * Uses a **Wav2Vec2** model from Hugging Face specifically trained to detect synthesized or cloned voices.
      * Features a robust audio extraction and preprocessing pipeline.
  * **📊 Interactive Dashboard**:
      * Displays the processed video side-by-side with the original.
      * Presents overall "Real" and "Fake" confidence scores with progress bars.
      * Includes a line chart to visualize the confidence fluctuations across all video frames.
  * **Efficient Processing**: Automatically trims uploaded videos to the first 10 seconds to ensure quick analysis and a responsive user experience.
  * **Automatic Cleanup**: Manages temporary files and folders, cleaning them up after each session.

-----

## Technology Stack

  * **Backend**: Python
  * **Frontend**: Streamlit
  * **Deep Learning**: PyTorch
  * **Core Models**:
      * **Video**: `facenet-pytorch` (MTCNN, InceptionResnetV1)
      * **Audio**: `transformers` (Hugging Face - Wav2Vec2)
      * **Visualization**: `pytorch-grad-cam`, `MediaPipe`
  * **Media Processing**: OpenCV, MoviePy, Librosa
  * **External Dependencies**: FFmpeg (for audio extraction and video processing)

-----

## Installation and Setup

Follow these steps to get the application running on your local machine.

### 1\. Prerequisites

  * **Python 3.9+**
  * **Git**
  * **FFmpeg**: You must have FFmpeg installed on your system, as the application relies on it for audio processing and video manipulation. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

### 2\. Clone the Repository

```bash
git clone https://your-repository-url.git
cd deepfake-detection-system
```

### 3\. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4\. Install Dependencies

A `requirements.txt` file should be created in the project root with the following content:

**`requirements.txt`**

```
streamlit
torch
torchvision
facenet-pytorch
numpy
pandas
opencv-python
pytorch-grad-cam
mediapipe
transformers
moviepy
librosa
soundfile
```

Install all the required libraries using pip:

```bash
pip install -r requirements.txt
```

### 5\. Download Model Checkpoint

The video classification model requires the `resnetinceptionv1_epoch_32.pth` file. Make sure you have this file in the root directory of the project.

### 6\. **IMPORTANT**: Configure FFmpeg Path

The script contains hardcoded paths to the FFmpeg executables. You **must** update these paths to match your system's installation location.

  * In `video11.py`, find the `run_ffmpeg` function and update the path to `ffmpeg.exe`.
  * In `audio.py` and `video11.py`, find the `has_audio` function and update the path to `ffprobe.exe`.

**Example (`video11.py`):**

```python
# Find this line and change the path
ffmpeg_command = [
    r"C:\path\to\your\ffmpeg\bin\ffmpeg.exe", 
    ...
]
```

-----

## Usage

Once the setup is complete, you can run the application with a single command:

```bash
streamlit run main.py
```

This will launch the Streamlit application in your default web browser. From there, you can upload an `.mp4` file and start the classification.

-----

## Project Structure

```
.
├── 📁 tempvid/
├── 📁 tempaudio/
├── 📁 pics/
├── ... (other temporary folders)
│
├── 📜 main.py             # Main Streamlit application file (UI and orchestration)
├── 📜 video.py            # Handles all video processing and classification
├── 📜 audio.py            # Handles all audio extraction and classification
│
├── 📜 requirements.txt    # List of Python dependencies
├── 📜 resnetinceptionv1_epoch_32.pth # Video classification model weights
└── 📜 README.md           # This file
```

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

  * This project relies on fantastic open-source models from [Hugging Face](https://huggingface.co/) and the deep learning community.
  * Special thanks to the developers of Streamlit, PyTorch, and FFmpeg for their invaluable tools.
