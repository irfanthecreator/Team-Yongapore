import streamlit as st
import cv2
import numpy as np
from openvino.runtime import Core
from tempfile import NamedTemporaryFile

# Define the vehicle detection model path
VEHICLE_DETECTION_MODEL = "models/vehicle-detection-adas-0002.xml"

def preprocess(frame, net_input_shape):
    """Resize and preprocess the input frame for inference."""
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))  # Convert HWC to CHW
    p_frame = p_frame.reshape(1, *p_frame.shape)  # Add batch dimension
    return p_frame

def calculate_distance(bbox_height, known_vehicle_height=1.5, focal_length=615):
    """
    Estimate the distance of the vehicle based on the bounding box height, a known vehicle height, and focal length.
    :param bbox_height: The height of the bounding box in pixels
    :param known_vehicle_height: The approximate real-world height of the vehicle (in meters)
    :param focal_length: The focal length of the camera (in pixels)
    :return: Estimated distance to the vehicle (in meters)
    """
    if bbox_height == 0:
        return None  # To avoid division by zero
    distance = (known_vehicle_height * focal_length) / bbox_height
    return distance

def draw_boxes(frame, output, width, height, distances, confidence_threshold=0.5):
    """
    Draw bounding boxes and estimate distance to detected vehicles.
    Change the box color to red if vehicles are within 3 meters.
    """
    focal_length = 615  # Adjust based on your camera
    known_vehicle_height = 1.5  # Approximate height of a vehicle in meters
    proximity_threshold = 3.0  # Distance threshold for color change in meters

    for box in output[0][0]:
        conf = box[2]
        if conf >= confidence_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            bbox_height = ymax - ymin  # Height of the bounding box

            # Calculate the distance to the vehicle
            distance = calculate_distance(bbox_height, known_vehicle_height, focal_length)
            distances.append(distance)

            # If the distance is less than the proximity threshold, color the box red; otherwise, blue
            if distance < proximity_threshold:
                box_color = (0, 0, 255)  # Red
            else:
                box_color = (255, 0, 0)  # Blue

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 3)
            label = f'Dist: {distance:.2f}m'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    return frame

def infer_on_video(input_file, confidence_threshold=0.5, device='CPU'):
    """Inference function for vehicle detection using OpenVINO."""
    core = Core()
    
    # Load and compile the vehicle detection model
    model = core.read_model(model=VEHICLE_DETECTION_MODEL)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)

    # Open video capture
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Streamlit placeholder for video frames
    stframe = st.empty()

    # Initialize the list to store distances
    distances = []  # Make sure it's a list, not a tuple

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame and run inference
        p_frame = preprocess(frame, input_layer.shape)
        results = compiled_model([p_frame])
        output = results[compiled_model.output(0)]

        # Draw bounding boxes for detected vehicles and calculate distances
        frame = draw_boxes(frame, output, width, height, distances, confidence_threshold)

        # Display the processed frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

# Streamlit UI layout
st.title("Vehicle Detection using OpenVINO")

# Upload video file input
input_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

# Confidence threshold slider
confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Device selection
device = st.selectbox("Select Device", ["CPU", "GPU"], index=0)

# Run inference when button is pressed
if st.button("Run Inference"):
    if input_video:
        # Save uploaded video to a temporary file for processing
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(input_video.read())
            temp_file_path = temp_file.name
        # Run vehicle detection on uploaded video
        infer_on_video(temp_file_path, confidence_threshold, device)
