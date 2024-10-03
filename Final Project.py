import streamlit as st
from ultralytics import YOLO
import re
import cv2
import numpy as np
import os
import threading
import time
import subprocess


# Make use of whole screen. (Left Aligned)
def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()


# Function to predict with the chosen model
def predict(chosen_model, img, classes=[], conf=0.5):
    # Check if classes are specified for filtering
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

# Function to draw bounding boxes and class names on the image
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=1, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    
    # Loop over detected results
    for result in results:
        # Loop over each detected box in the result
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            
            # Draw rectangle for bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
            
            # Draw class label on the image
            class_name = result.names[int(box.cls[0])]
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    
    return img, results


def convert_video(input_video_path, output_video_path):
    try:
        # Run the ffmpeg command to convert the video
        ffmpeg_command = [
            'ffmpeg', '-i', input_video_path,
            '-vcodec', 'libx264',  # Specify the H.264 codec
            output_video_path
        ]
        
        # Use subprocess to execute the command
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the conversion was successful
        if result.returncode == 0:
            print(f"Video conversion successful: {output_video_path}")
            return output_video_path
        else:
            print(f"Video conversion failed: {result.stderr.decode('utf-8')}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Designing Sidebar
sideb = st.sidebar

with sideb:
    # st.title("")

    st.header("Task")
    task = st.selectbox(
        'Select Task:',
        ('Detection', 'Segmentation'),
        index=0) 

    st.header("YOLO Version")
    yolo_version = st.selectbox(
        'Select YOLO Version:',
        ('YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLOv11'),
        index=1) 
    
    # Regex pattern to get the last digits
    pattern = r'(\d+)\D*$'
    match = re.search(pattern, yolo_version)
    if match:
        version_num = match.group(1)

    version_num = int(version_num)


    if task == "Detection":
        if version_num == 8:
            models = ('YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x')
        elif version_num == 9:
            models = ('YOLOv9t', 'YOLOv9s', 'YOLOv9m', 'YOLOv9c', 'YOLOv9e')
        elif version_num == 10:
            models = ('YOLOv10n', 'YOLOv10s', 'YOLOv10m', 'YOLOv10l', 'YOLOv10x')
        elif version_num == 11:
            models = ('YOLO11n', 'YOLO11s', 'YOLO11m', 'YOLO11l', 'YOLO11x')

    elif task == "Segmentation":
        if version_num == 8:
            models = ('YOLOv8n-Seg', 'YOLOv8s-Seg', 'YOLOv8m-Seg', 'YOLOv8l-Seg', 'YOLOv8x-Seg')
        elif version_num == 9:
            models = ('YOLOv9n-Seg', 'YOLOv9s-Seg', 'YOLOv9m-Seg', 'YOLOv9l-Seg', 'YOLOv9x-Seg')
        elif version_num == 10:
            models = ('YOLOv10n-Seg', 'YOLOv10s-Seg', 'YOLOv10m-Seg', 'YOLOv10l-Seg', 'YOLOv10x-Seg') 
        elif version_num == 11:
            models = ('YOLOv11n-Seg', 'YOLOv11s-Seg', 'YOLOv11m-Seg', 'YOLOv11l-Seg', 'YOLOv11x-Seg')
    
    else:
        models = ('YOLOv9n', 'YOLOv9s', 'YOLOv9m', 'YOLOv9l', 'YOLOv9x')

    # print(models)

    st.header("Model")
    if models:
        yolo_model = st.selectbox(
            'Select YOLO Model:',
            models,
            index=0)   
    else:
        yolo_model = st.selectbox(
            'Select YOLO Model:',
            (),
            index=0)
        
    st.header("Device")
    device = st.selectbox(
        'Select Preferred Device:',
        ('CPU', 'GPU'),
        index=1)   
    
    st.header("Input Type")
    input_type = st.selectbox(
        'Select Input Type:',
        ('Image', 'Video'),
        index=0)   


    st.session_state.download_status = False


    if os.path.exists(f'/tf/{yolo_model.lower()}.pt'):
        model = YOLO(f'{yolo_model.lower()}')
        st.session_state.model_status = f'<p style="color:green;">Model {yolo_model} loaded successfully.</p>'
        st.session_state.download_text = ""
    else:
        model = None
        st.session_state.model_status = f'<p style="color:red;">Model {yolo_model} not found.</p>'
        

    st.markdown(st.session_state.model_status, unsafe_allow_html=True)


    st.markdown('**Developed By:** M. Raaid Khan')



# Header    
st.header("YOLOvX Test Bench")
st.write("An Application to test YOLO Models for Aerial Object Detection and Image Segmentation in Pictures and Videos")


col1, col2 = st.columns([3, 1])

def resize_image(image, height):
    aspect_ratio = image.shape[1] / image.shape[0]  # width / height
    new_width = int(height * aspect_ratio)
    return cv2.resize(image, (new_width, height))

# Image/Video Area
with col1:
     
    if model is not None:
        if input_type == "Image":

            uploaded_image = st.file_uploader('', type='jpg', key=6)


            if uploaded_image is not None:
                file_bytes = uploaded_image.read()
                org_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                resized_image = resize_image(org_image, 400)
                org_image = resized_image.copy()

                
                result_img, _ = predict_and_detect(model, resized_image, classes=[], conf=0.5)

                left_col, right_col= st.columns(2)

                with left_col:
                    st.image(org_image, channels="BGR", caption="Original Image",)

                with right_col:
                    st.image(result_img, channels="BGR", caption="Predicted Image",)

            output_path = "/tf/output_video.mp4"
            st.video(output_path)

        if input_type == "Video":     
            # Upload a video file
            uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

            if uploaded_video is not None:
                file_binary = uploaded_video.read()
                input_path = uploaded_video.name
                print(input_path)

                with open(input_path, "wb") as temp_file:
                    temp_file.write(file_binary)

                video_stream = cv2.VideoCapture(input_path)

                width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
                height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
                fps = int(video_stream.get(cv2.CAP_PROP_FPS)) 

                # Create a VideoWriter object

                output_path = "/tf/output_video.mp4"

                out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                rectangle_thickness = 1
                text_thickness = 1

                with st.spinner('Processing video...'): 
                    while True:
                        ret, frame = video_stream.read()
                        if not ret:
                            break
                        result = model(frame)
                        for detection in result[0].boxes.data:
                            x0, y0 = (int(detection[0]), int(detection[1]))
                            x1, y1 = (int(detection[2]), int(detection[3]))
                            score = round(float(detection[4]), 2)
                            cls = int(detection[5])
                            object_name =  model.names[cls]
                            label = f'{object_name} {score}' 

                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), rectangle_thickness)
                            cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), text_thickness)

                        detections = result[0].verbose()
                        cv2.putText(frame, detections, (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                        out_video.write(frame) 

                    video_stream.release()
                    out_video.release()

                st.write("Video Processed Successfully")

                print("Converting Video...")
                converted_video_path = convert_video(output_path, 'output_video_that_streamlit_can_play.mp4') #Delete Old Video or Change Name

                st.video(converted_video_path)
            

    else:
        st.error("Please select a Valid model first.")
