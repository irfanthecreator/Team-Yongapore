import cv2
import streamlit as st
from openvino.runtime import Core
import numpy as np

# Define the models
VEHICLE_DETECTION_MODEL = "models/vehicle-detection-adas-0002.xml"

def preprocess(frame, net_input_shape):
    """Resize, transpose, and prepare the input frame for model inference."""
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame

def calculate_distance(bbox_height, known_vehicle_height=1.5, focal_length=615):
    """Estimate the distance of the vehicle based on the bounding box height."""
    if bbox_height == 0:
        return None
    distance = (known_vehicle_height * focal_length) / bbox_height
    return distance

def draw_boxes(frame, output, width, height, confidence_threshold=0.5):
    """Draw bounding boxes and estimate distance to detected vehicles."""
    focal_length = 615
    known_vehicle_height = 1.5
    proximity_threshold = 3.0  # 3 meters threshold
    vehicle_count = 0

    for box in output[0][0]:
        conf = box[2]
        if conf >= confidence_threshold:
            vehicle_count += 1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            bbox_height = ymax - ymin

            # Calculate the distance to the vehicle
            distance = calculate_distance(bbox_height, known_vehicle_height, focal_length)

            # If the distance is less than the proximity threshold, color the box red; otherwise, blue
            box_color = (0, 0, 255) if distance < proximity_threshold else (255, 0, 0)

            # Draw the bounding box and distance label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 3)
            label = f'Dist: {distance:.2f}m'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    return frame, vehicle_count

def infer_on_video(input_file, confidence_threshold=0.5, device='CPU'):
    """Inference function for vehicle detection using OpenVINO."""
    core = Core()
    model = core.read_model(VEHICLE_DETECTION_MODEL)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)

    # Open video capture
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    stframe = st.empty()  # Streamlit placeholder for video frames
    progress_bar = st.progress(0)  # Streamlit progress bar

    # Process video frame by frame
    processed_frames = 0
    total_vehicle_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame and run inference
        p_frame = preprocess(frame, input_layer.shape)
        results = compiled_model([p_frame])
        output = results[compiled_model.output(0)]

        # Draw bounding boxes for detected vehicles and calculate distances
        frame, vehicle_count = draw_boxes(frame, output, width, height, confidence_threshold)
        total_vehicle_count += vehicle_count

        # Display the processed frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Update progress bar
        processed_frames += 1
        progress_bar.progress(int(processed_frames / total_frames * 100))

    cap.release()

    # Display vehicle count after processing is finished
    st.success(f"Total vehicles detected: {total_vehicle_count}")

# Streamlit UI
st.title("AI Vehicle Detection with OpenVINO")
st.sidebar.title("Settings")

# Problem Scoping (4W's)
with st.expander("Project Description: Problem Scoping (4W's)"):
    st.markdown("""
    **Who (누구):**  
    The users facing this problem are drivers who aim to enhance safety while navigating roadways.
    
    **What (무엇):**  
    While drivers are generally aware of their surroundings, they can encounter situations where it is difficult to fully perceive potential hazards—especially in complex or busy environments. This can increase the risk of accidents, as certain dangers might not be easily noticeable. 

    **Where (어디):**  
    This issue arises in various driving environments such as urban areas, highways, intersections, and parking lots, where vehicle and pedestrian interactions create complex situations.

    **Why (왜):**  
    Solving this problem improves drivers' situational awareness, reduces accident rates, and enhances overall road safety.
    """)

# Upload video
input_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi"])

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Device selection (default is CPU)
device = st.sidebar.selectbox("Select Device", ["CPU", "GPU"], index=0)

# Run inference button
if st.sidebar.button("Run Inference") and input_video:
    with st.spinner("Processing..."):
        # Save uploaded video to a temporary file for processing
        with open("uploaded_video.mp4", "wb") as f:
            f.write(input_video.read())
        # Run the inference
        infer_on_video("uploaded_video.mp4", confidence_threshold, device)
