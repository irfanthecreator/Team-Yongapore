import cv2
import streamlit as st
from openvino.runtime import Core
import numpy as np

# Define the models
VEHICLE_DETECTION_MODEL = "models/vehicle-detection-adas-0002.xml"

def preprocess(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

def calculate_distance(bbox_height, known_vehicle_height=1.5, focal_length=615):
    if bbox_height == 0:
        return None
    distance = (known_vehicle_height * focal_length) / bbox_height
    return distance

def draw_boxes(frame, output, width, height, confidence_threshold=0.5):
    focal_length = 615
    known_vehicle_height = 1.5
    proximity_threshold = 3.0
    
    for box in output[0][0]:
        conf = box[2]
        if conf >= confidence_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            bbox_height = ymax - ymin
            distance = calculate_distance(bbox_height, known_vehicle_height, focal_length)
            
            box_color = (0, 0, 255) if distance < proximity_threshold else (255, 0, 0)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 3)
            label = f'Dist: {distance:.2f}m'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return frame

def infer_on_video(input_file, confidence_threshold=0.5, device='CPU'):
    core = Core()
    model = core.read_model(VEHICLE_DETECTION_MODEL)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    stframe = st.empty()

    frame_skip = 5  # Only process every 5th frame
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Skip frames to improve performance
        if frame_counter % frame_skip == 0:
            p_frame = preprocess(frame, input_layer.shape)
            results = compiled_model([p_frame])
            output = results[compiled_model.output(0)]
            frame = draw_boxes(frame, output, width, height, confidence_threshold)

        stframe.image(frame, channels="BGR", use_column_width=True)
        frame_counter += 1


# Streamlit UI
st.title("AI Video Inference using OpenVINO")
input_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
device = st.selectbox("Select Device", ["CPU", "GPU"], index=0)

if st.button("Run Inference") and input_video:
    with st.spinner("Processing..."):
        with open("uploaded_video.mp4", "wb") as f:
            f.write(input_video.read())
        infer_on_video("uploaded_video.mp4", confidence_threshold, device)
