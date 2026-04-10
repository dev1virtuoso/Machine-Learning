import cv2
import mediapipe as mp
import numpy as np
import json
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.hub import load

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# MiDaS depth estimation setup
def load_midas_model():
    """
    Load the MiDaS depth estimation model.
    """
    print("Loading MiDaS depth estimation model...")
    model = load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()  # Set model to evaluation mode
    return model

midas_model = load_midas_model()
transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

# Updated Landmarks Without Facial Features
pose_tubuh = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

pose_tangan = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP',
    'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP',
    'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
    'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

pose_tangan_2 = [name + '2' for name in pose_tangan]  # For left hand

# Depth estimation function
def estimate_depth(frame, model, transform):
    """
    Perform depth estimation using MiDaS.
    """
    height, width = frame.shape[:2]
    if height % 32 != 0 or width % 32 != 0:
        # Resize frame to ensure dimensions are divisible by 32
        frame = cv2.resize(frame, ((width // 32) * 32, (height // 32) * 32))
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb_frame).unsqueeze(0)

    with torch.no_grad():
        depth_map = model(input_batch).squeeze().cpu().numpy()

    # Normalize depth map
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return depth_map

# Extract landmarks and compute depth
def extract_landmarks_with_depth(results, depth_map, image_shape, frame_count):
    """
    Extract pose and hand landmarks with depth data.
    """
    frame_data = {"frame": frame_count, "landmarks": {}}
    has_landmarks = False

    if depth_map.shape[:2] != (image_shape[0], image_shape[1]):
        depth_map = cv2.resize(depth_map, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Pose landmarks
    if results.pose_landmarks:
        frame_data["landmarks"]["pose"] = {
            pose_tubuh[i]: (
                lm.x * image_shape[1],
                lm.y * image_shape[0],
                depth_map[
                    min(int(lm.y * image_shape[0]), depth_map.shape[0] - 1),
                    min(int(lm.x * image_shape[1]), depth_map.shape[1] - 1)
                ]
            )
            for i, lm in enumerate(results.pose_landmarks.landmark)
            if i < len(pose_tubuh)  # Process only non-facial pose landmarks
        }
        has_landmarks = True

    # Right hand landmarks
    if results.right_hand_landmarks:
        frame_data["landmarks"]["right_hand"] = {
            pose_tangan[i]: (
                lm.x * image_shape[1],
                lm.y * image_shape[0],
                depth_map[
                    min(int(lm.y * image_shape[0]), depth_map.shape[0] - 1),
                    min(int(lm.x * image_shape[1]), depth_map.shape[1] - 1)
                ]
            )
            for i, lm in enumerate(results.right_hand_landmarks.landmark)
            if i < len(pose_tangan)
        }
        has_landmarks = True

    # Left hand landmarks
    if results.left_hand_landmarks:
        frame_data["landmarks"]["left_hand"] = {
            pose_tangan_2[i]: (
                lm.x * image_shape[1],
                lm.y * image_shape[0],
                depth_map[
                    min(int(lm.y * image_shape[0]), depth_map.shape[0] - 1),
                    min(int(lm.x * image_shape[1]), depth_map.shape[1] - 1)
                ]
            )
            for i, lm in enumerate(results.left_hand_landmarks.landmark)
            if i < len(pose_tangan_2)
        }
        has_landmarks = True

    if has_landmarks:
        processed_data.append(frame_data)

# Process video without displaying results
video_path = input("Enter the MP4 file path: ")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video. Check the file path.")
    exit()

processed_data = []  # To store all frames with valid landmark data
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and preprocess frame
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)

        # Estimate depth and extract landmarks
        depth_map = estimate_depth(frame, midas_model, transform)
        extract_landmarks_with_depth(results, depth_map, frame.shape, frame_count)

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames} ({(frame_count / total_frames) * 100:.2f}%)")

cap.release()

# Custom function to convert non-serializable objects into JSON-serializable formats
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to standard Python lists
    if isinstance(obj, np.uint8):
        return int(obj)  # Convert uint8 objects to integers
    return obj  # Leave other objects unchanged

# Save JSON data
if processed_data:
    with open("coordinate_with_depth.json", "w") as f:
        json.dump(processed_data, f, indent=4, default=convert_to_serializable)
    print("Landmark data with depth saved to coordinate_with_depth.json.")
else:
    print("No valid data found to save.")
