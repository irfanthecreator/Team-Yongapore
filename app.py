import cv2
import streamlit as st
from openvino.runtime import Core
import numpy as np
import time

# Define the vehicle detection model
VEHICLE_DETECTION_MODEL = "models/vehicle-detection-adas-0002.xml"

# Preprocess the video frame for model inference
def preprocess(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

# Draw bounding boxes on detected vehicles and calculate distance
def draw_boxes(frame, output, width, height, confidence_threshold=0.5, box_color=(255, 0, 0)):
    focal_length = 615  # Focal length for distance estimation
    known_vehicle_height = 1.5  # Average height of a vehicle in meters
    proximity_threshold = 3.0  # Distance threshold for proximity warning
    
    for box in output[0][0]:
        conf = box[2]
        if conf >= confidence_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            bbox_height = ymax - ymin
            distance = (known_vehicle_height * focal_length) / bbox_height if bbox_height > 0 else None
            
            # Change box color based on proximity
            box_color = (0, 0, 255) if distance and distance < proximity_threshold else box_color
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 3)
            label = f'Distance: {distance:.2f}m' if distance else 'N/A'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return frame

# Perform inference on the video and generate the output
def infer_on_video(input_file, confidence_threshold=0.5, box_color=(255, 0, 0), device='CPU'):
    core = Core()
    model = core.read_model(VEHICLE_DETECTION_MODEL)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Save the processed frames into a new video file
    output_file = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))  # 30 FPS
    
    # Frame-by-frame progress bar
    progress_bar = st.progress(0)
    frame_num = 0
    
    # Time measurement
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame and run inference
        p_frame = preprocess(frame, input_layer.shape)
        results = compiled_model([p_frame])
        output = results[compiled_model.output(0)]
        
        frame = draw_boxes(frame, output, width, height, confidence_threshold, box_color)
        
        # Write the processed frame to the output video
        out.write(frame)
        
        # Update progress
        frame_num += 1
        progress_bar.progress(frame_num / total_frames)
    
    # Measure FPS
    elapsed_time = time.time() - start_time
    fps = total_frames / elapsed_time
    st.write(f"Processing Time: {elapsed_time:.2f}s, FPS: {fps:.2f}")
    
    cap.release()
    out.release()

    return output_file

# Streamlit UI
st.title("AI Vehicle Detection with Advanced Features")

# Video uploader for Streamlit
input_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

# Confidence threshold slider
confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Bounding box color toggle
box_color_option = st.selectbox("Bounding Box Color", ["Red", "Blue", "Green"], index=1)
box_color = (255, 0, 0) if box_color_option == "Blue" else (0, 255, 0) if box_color_option == "Green" else (0, 0, 255)

# Device selection
device = st.selectbox("Select Device", ["CPU", "GPU"], index=0)

if st.button("Run Inference") and input_video:
    with st.spinner("Processing..."):
        # Save the uploaded video locally
        with open("uploaded_video.mp4", "wb") as f:
            f.write(input_video.read())
        
        # Run inference on the uploaded video
        output_video = infer_on_video("uploaded_video.mp4", confidence_threshold, box_color, device)
        
        # Display the processed video
        video_file = open(output_video, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
