import cv2
import streamlit as st
from openvino.runtime import Core
import numpy as np
import requests
from tempfile import NamedTemporaryFile

# Define the vehicle detection model
VEHICLE_DETECTION_MODEL = "models/vehicle-detection-adas-0002.xml"

# Preprocess the video frame for model inference
def preprocess(frame, net_input_shape):
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

# Draw bounding boxes on detected vehicles and calculate distance
def draw_boxes(frame, output, width, height, confidence_threshold=0.5):
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
            box_color = (0, 0, 255) if distance and distance < proximity_threshold else (255, 0, 0)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 3)
            label = f'Distance: {distance:.2f}m' if distance else 'N/A'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return frame

# Download video from URL
def download_video(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        temp_video.write(chunk)
                return temp_video.name
        else:
            st.error("Failed to download video from URL.")
            return None
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

# Perform inference on the video and generate the output
def infer_on_video(input_file, confidence_threshold=0.5, device='CPU'):
    core = Core()
    model = core.read_model(VEHICLE_DETECTION_MODEL)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Save the processed frames into a new video file
    output_file = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))  # 30 FPS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame and run inference
        p_frame = preprocess(frame, input_layer.shape)
        results = compiled_model([p_frame])
        output = results[compiled_model.output(0)]
        
        frame = draw_boxes(frame, output, width, height, confidence_threshold)
        
        # Write the processed frame to the output video
        out.write(frame)
    
    cap.release()
    out.release()

    return output_file

# Streamlit UI
st.title("AI Vehicle Detection using OpenVINO")

# Option to upload a video or use a URL
input_option = st.selectbox("Select Video Source", ["Upload a video", "Use video URL"])

if input_option == "Upload a video":
    input_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    video_url = None
elif input_option == "Use video URL":
    video_url = st.text_input("Enter video URL (mp4 format)")
    input_video = None

confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
device = st.selectbox("Select Device", ["CPU", "GPU"], index=0)

if st.button("Run Inference"):
    with st.spinner("Processing..."):
        if input_video:
            # Save the uploaded video locally
            with open("uploaded_video.mp4", "wb") as f:
                f.write(input_video.read())
            video_path = "uploaded_video.mp4"
        elif video_url:
            # Download the video from the URL
            video_path = download_video(video_url)
            if not video_path:
                st.error("Failed to download video.")
                st.stop()
        
        # Run inference on the selected video
        output_video = infer_on_video(video_path, confidence_threshold, device)

        # Display the processed video
        video_file = open(output_video, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
