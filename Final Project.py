import streamlit as st

# Make use of whole screen. (Left Aligned)
def wide_space_default():
    st.set_page_config(page_title="YOLOvX Test Bench",layout="wide")

wide_space_default()

from ultralytics import YOLO
import torch
import re
import cv2
import numpy as np
import os
import threading
import time
import subprocess
import pandas as pd


# Function to predict with the chosen model
def predict(chosen_model, img, classes=[], conf=0.5):
    # Check if classes are specified for filtering
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

# Function to draw bounding boxes and class names on the image
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=1, text_thickness=1, font_scale=1):
    results = predict(chosen_model, img, classes, conf=conf)
    
    print("Making Boxes With....")
    print(f"Rectangle Thickness: {rectangle_thickness}")
    print(f"Text Thickness: {text_thickness}")

    # Loop over detected results
    for result in results:
        # Loop over each detected box in the result
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            
            # Draw rectangle for bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), rectangle_thickness)
            
            # Draw class label on the image
            class_name = result.names[int(box.cls[0])]  #(Class Name Only)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 0), text_thickness)
    
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

# <---Designing Sidebar--->
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
    device_selection = st.selectbox(
        'Select Preferred Device:',
        ('CPU', 'GPU'),
        index=1)   
    
    st.header("Input Type")
    input_type = st.selectbox(
        'Select Input Type:',
        ('Image', 'Video'),
        index=0)   
    

    if os.path.exists(f'/tf/{yolo_model.lower()}.pt'):
        model = YOLO(f'{yolo_model.lower()}')
        st.session_state.model_status = f'<p style="color:green;">Model {yolo_model} loaded successfully.</p>'

        if device_selection == 'GPU':
            if torch.cuda.is_available():
                print("CUDA is available; Using GPU.")
                model.to(torch.device('cuda'))
        elif device_selection == 'CPU':
           model.to(torch.device('cpu'))
           print("Using CPU.")
    else:
        model = None
        st.session_state.model_status = f'<p style="color:red;">Model {yolo_model} not found.</p>'
        

    st.markdown(st.session_state.model_status, unsafe_allow_html=True)


    st.markdown('**Developed By:** M. Raaid Khan')

# <---Sidebar Design Ends--->


# <---Designing Main Area--->

col1, col2 = st.columns([3, 1])

def resize_image(image, height):
    aspect_ratio = image.shape[1] / image.shape[0]  # width / height
    new_width = int(height * aspect_ratio)
    return cv2.resize(image, (new_width, height))


# Image/Video Area

with col2:
    st.subheader("Paramters")
    
    st.session_state.conf = st.slider("Confidence", 0.0, 1.0, 0.3,)       
    st.session_state.rectangle_thickness = st.slider("Rectangle Thickness", 1, 5, 1,)
    st.session_state.text_thickness = st.slider("Text Thickness", 1, 5, 1,)
    st.session_state.font_scale = st.slider("Text Size", 0.5, 1.5, 1.0,)
    st.session_state.show_original = st.toggle("Show Original Image", False)

with col1:
    # Header    
    st.header("YOLOvX Test Bench")
    st.write("An Application to test YOLO Models for Object Detection and Image Segmentation in Pictures and Videos")

    # This function will only run once when the app starts
    def app_startup():
        global org_image
        global result_img
        org_image = None
        result_img = None


        st.session_state['initialized'] = True

        # Setting Defualt Parameters
        if 'conf' not in st.session_state:
            st.session_state.conf = 0.3
        if 'text_thickness ' not in st.session_state:
            st.session_state.text_thickness = 1
        if 'rectangle_thickness' not in st.session_state:
            st.session_state.rectangle_thickness = 1
        if 'show_original' not in st.session_state:
            st.session_state.show_original = False
        if 'font_scale' not in st.session_state:
            st.session_state.font_scale = 1

        # Check to Display Summary if Image Detection Model is Executed 
        if 'image_model_executed' not in st.session_state:
            st.session_state.image_model_executed = False

    # Check if the app has been initialized
    if 'initialized' not in st.session_state:
        app_startup()

    if model is not None:
        if input_type == "Image":

            uploaded_image = st.file_uploader('', type=['jpg','png'], key=6)

            if uploaded_image is not None:
                file_bytes = uploaded_image.read()
                org_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                resized_image = resize_image(org_image, 400)
                org_image = resized_image.copy()

                start_time = time.time()

                result_img, model_results = predict_and_detect(model, resized_image, classes=[], conf=st.session_state.conf,
                                                   rectangle_thickness=st.session_state.rectangle_thickness, 
                                                   text_thickness=st.session_state.text_thickness,
                                                   font_scale=st.session_state.font_scale)
                
                end_time = time.time()
                inference_time = end_time - start_time

                if st.session_state.show_original == True:
                    left_col, right_col= st.columns(2)
                    with left_col:
                        st.image(org_image, channels="BGR", caption="Original Image",)

                    with right_col:
                        st.image(result_img, channels="BGR", caption="Predicted Image",)

                else:
                    left_col, mid_col, right_col = st.columns([1,3,1])
                    with mid_col:
                        st.image(result_img, channels="BGR", caption="Predicted Image",)

                st.session_state.image_model_executed = True


        if input_type == "Video":   
            model_results = None  
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
                # Video Created by OpenCV with Bounding Boxes
                output_path = "/tf/output_video.mp4"
                out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                

                with st.spinner('Processing video...'): 
                    while True:
                        ret, frame = video_stream.read()
                        if not ret:
                            break
                        result = model(frame, conf=st.session_state.conf)
                        for detection in result[0].boxes.data:
                            x0, y0 = (int(detection[0]), int(detection[1]))
                            x1, y1 = (int(detection[2]), int(detection[3]))
                            score = round(float(detection[4]), 2)
                            cls = int(detection[5])
                            object_name =  model.names[cls]
                            label = f'{object_name} {score}' 

                            # Putting Text and Rectangle on Each Object
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), st.session_state.rectangle_thickness)
                            cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, st.session_state.font_scale, (255, 0, 0), st.session_state.text_thickness)

                        detections = result[0].verbose()

                        #Putting Text on Top Left with Frame Summary
                        cv2.putText(frame, detections, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, (st.session_state.font_scale+0.5), (0, 255, 0), (st.session_state.text_thickness+1))

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                        out_video.write(frame) 

                    video_stream.release()
                    out_video.release()

                st.write("Video Processed Successfully")

                print("Converting Video...")
                file_name = os.path.splitext(uploaded_video.name)[0]
                converted_video_path = convert_video(output_path, f'{file_name}_{time.time()}.mp4') #Delete Old Video or Change Name

                st.video(converted_video_path)
    
    else:
        st.error("Please select a Valid model first.")

def detections_to_dataframe(results, class_names, thres):
        thres = thres / 100
    	# Initialize lists to store detection data
        all_boxes = []
        all_confidences = []
        all_class_ids = []
        all_class_names = []

    	# Iterate over each detection in the results
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].item())  # Extract class ID
            class_name = results[0].names[int(box.cls[0])]
            conf = round(box.conf[0].item(), 2)  # Extract confidence score

            if conf >= thres:  # Apply threshold
                    all_boxes.append([xmin, ymin, xmax, ymax])
                    all_confidences.append(conf)
                    all_class_ids.append(cls_id)
                    all_class_names.append(class_name)  # Convert class ID to class name

        # Create a DataFrame
        df = pd.DataFrame(all_boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        df['confidence'] = all_confidences
        df['class'] = all_class_ids
        df['name'] = all_class_names

        return df

with col2:
    if st.session_state.image_model_executed:
        df = detections_to_dataframe(model_results, class_names=[], thres= st.session_state.conf)


        summary = {
            "Total Objects Detected": len(df),
            "Classes Detected": df['name'].unique().tolist(),
            "Confidence Scores": df['confidence'].tolist(),
            "Bounding Boxes": df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist(),
        }

         # Display the summary
        st.write("### Summary of Detection")
        st.write(f"**Total Objects Detected:** {summary['Total Objects Detected']}")
        st.write(f"**Classes Detected:** {', '.join(summary['Classes Detected'])}")
        st.write(f"**Confidence Scores:** {summary['Confidence Scores']}")
        # st.write("**Bounding Boxes:**")
        # st.write(summary['Bounding Boxes'])

        # Time taken for inference
        st.write(f"**Inference Time:** {inference_time:.2f} Seconds")

        st.session_state.image_model_executed = False