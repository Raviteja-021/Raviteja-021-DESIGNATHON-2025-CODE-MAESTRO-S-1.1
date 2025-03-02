import streamlit as st
import os
import numpy as np
import pandas as pd
from audio import extract_audio, classify_audio, has_audio
from video import run, removefilesinfold, trim_video
from moviepy.video.io.VideoFileClip import VideoFileClip

def save_uploaded_file(uploaded_file):
    save_path = os.path.join("tempvid", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def onchange():
    removefilesinfold("tempvid")
    removefilesinfold("temppics")
    removefilesinfold("tempaudio")
    removefilesinfold("output_of_photo")
    removefilesinfold("boxpics")
    removefilesinfold("pics")
    removefilesinfold("mixdvid")
    removefilesinfold("vid")
    removefilesinfold("boxvid")

st.set_page_config(page_title='deepfake classification', layout='wide', initial_sidebar_state="collapsed")
st.write(" # $~$ Video:video_camera:")

with st.container(border=True):
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"], on_change=onchange)

col1, col2 = st.columns(2)
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    trimmed_video_path = "tempvid/trimmed_video.mp4"
    trim_video(file_path, trimmed_video_path)  # Trim the video to 10 seconds

    if has_audio(trimmed_video_path):
        audio = extract_audio(trimmed_video_path)
        with col1:
            with st.container(border=True):
                st.video(trimmed_video_path)
                st.audio(audio)
                aud_on_or_of = st.checkbox("Classify audio")
                classify = st.button("Classify")

        if aud_on_or_of and classify:
            confid, outvideo, showconf = run(trimmed_video_path)
            audconf = classify_audio(audio)
            with col2:
                with st.container(border=True):
                    st.video(outvideo)
                with st.container(border=True):
                    st.write(f"# Real: {float('{:.3f}'.format(showconf[0]))}%")
                    st.progress(showconf[0])
                    st.write(f"# Fake: {float('{:.3f}'.format(showconf[1]))}%")
                    st.progress(showconf[1])
                with st.container(border=True):
                    chart_data = pd.DataFrame(confid, columns=["real", "fake"])
                    st.line_chart(chart_data)

            st.write(f"## Audio results:")
            with st.container(border=True):
                st.write(f"# Real percentage={float('{:.3f}'.format(audconf[0][1]))}")
                st.progress(audconf[0][0] / 100)
                st.write(f"# Fake percentage={float('{:.3f}'.format(audconf[0][0]))}%")
                st.progress(audconf[0][1] / 100)

            # Remove temporary files after displaying the results
            removefilesinfold('tempvid')
            removefilesinfold('temppics')
            removefilesinfold('tempaudio')
            removefilesinfold('output_of_photo')
            removefilesinfold('boxpics')
            removefilesinfold('pics')
            removefilesinfold('mixdvid')
            removefilesinfold('vid')
            removefilesinfold('boxvid')
        elif classify:
            confid, outvideo, showconf = run(trimmed_video_path)
            with col2:
                with st.container(border=True):
                    st.video(outvideo)
                with st.container(border=True):
                    st.write(f"# Real: {float('{:.3f}'.format(showconf[0]))}%")
                    st.progress(showconf[0])
                    st.write(f"# Fake: {float('{:.3f}'.format(showconf[1]))}%")
                    st.progress(showconf[1])
                with st.container(border=True):
                    chart_data = pd.DataFrame(confid, columns=["real", "fake"])
                    st.line_chart(chart_data)

            # Remove temporary files after displaying the results
            removefilesinfold('tempvid')
            removefilesinfold('temppics')
            removefilesinfold('tempaudio')
            removefilesinfold('output_of_photo')
            removefilesinfold('boxpics')
            removefilesinfold('pics')
            removefilesinfold('mixdvid')
            removefilesinfold('vid')
            removefilesinfold('boxvid')
    else:
        with col1:
            with st.container(border=True):
                st.video(trimmed_video_path)
                classify = st.button("Classify")
        if classify:
            confid, outvideo, showconf = run(trimmed_video_path)
            with col2:
                with st.container(border=True):
                    st.video(outvideo)
                with st.container(border=True):
                    st.write(f"## Real: {float('{:.3f}'.format(showconf[0]))}%")
                    st.progress(showconf[0])
                    st.write(f"## Fake: {float('{:.3f}'.format(showconf[1]))}%")
                    st.progress(showconf[1])
                with st.container(border=True):
                    chart_data = pd.DataFrame(confid, columns=["real", "fake"])
                    st.line_chart(chart_data)

            # Remove temporary files after displaying the results
            removefilesinfold('tempvid')
            removefilesinfold('temppics')
            removefilesinfold('tempaudio')
            removefilesinfold('output_of_photo')
            removefilesinfold('boxpics')
            removefilesinfold('pics')
            removefilesinfold('mixdvid')
            removefilesinfold('vid')
            removefilesinfold('boxvid')