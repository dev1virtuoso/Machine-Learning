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
    model = load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()  # Set model to evaluation mode
    return model

midas_model = load_midas_model()
transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

# Variables for tracking
target_fps = 30
frame_duration = 1.0 / target_fps
processed_data = []  # To store all frames with valid landmark data

# Define body and hand landmarks for mapping
pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
              'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
              'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
              'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
              'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

pose_tangan = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
               'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
               'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

pose_tangan_2 = [name + '2' for name in pose_tangan]

# Depth estimation function
def estimate_depth(frame, model, transform):
    # Convert frame to RGB and preprocess
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb_frame).unsqueeze(0)

    # Perform depth estimation
    with torch.no_grad():
        depth_map = model(input_batch)
    depth_map = depth_map.squeeze().cpu().numpy()

    # Normalize depth map
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return depth_map

# Extract landmarks and compute depth
def extract_landmarks_with_depth(results, depth_map, image_shape, frame_count):
    frame_data = {"frame": frame_count, "landmarks": {}}
    has_landmarks = False

    # Pose landmarks
    if results.pose_landmarks:
        frame_data["landmarks"]["pose"] = {
            pose_tubuh[i]: (
                lm.x * image_shape[1],
                lm.y * image_shape[0],
                depth_map[int(lm.y * image_shape[0]), int(lm.x * image_shape[1])]  # Depth Z-coordinate
            )
            for i, lm in enumerate(results.pose_landmarks.landmark)
        }
        has_landmarks = True

    # Right hand landmarks
    if results.right_hand_landmarks:
        frame_data["landmarks"]["right_hand"] = {
            pose_tangan[i]: (
                lm.x * image_shape[1],
                lm.y * image_shape[0],
                depth_map[int(lm.y * image_shape[0]), int(lm.x * image_shape[1])]  # Depth Z-coordinate
            )
            for i, lm in enumerate(results.right_hand_landmarks.landmark)
        }
        has_landmarks = True

    # Left hand landmarks
    if results.left_hand_landmarks:
        frame_data["landmarks"]["left_hand"] = {
            pose_tangan_2[i]: (
                lm.x * image_shape[1],
                lm.y * image_shape[0],
                depth_map[int(lm.y * image_shape[0]), int(lm.x * image_shape[1])]  # Depth Z-coordinate
            )
            for i, lm in enumerate(results.left_hand_landmarks.landmark)
        }
        has_landmarks = True

    if has_landmarks:
        processed_data.append(frame_data)

# Process video using MediaPipe Holistic
video_path = input("Enter the MP4 file path: ")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video. Check the file path.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video stream or error reading frame.")
            break

        start_time = time.time()

        # Flip and preprocess frame
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = holistic.process(frame)

        # Estimate depth using MiDaS
        depth_map = estimate_depth(frame, midas_model, transform)

        # Extract landmarks and include depth
        extract_landmarks_with_depth(results, depth_map, frame.shape, frame_count)
        frame_count += 1

        # Display progress
        elapsed_time = time.time() - start_time
        print(f"Processed frame {frame_count}/{total_frames} ({(frame_count / total_frames) * 100:.2f}%)")

        # Show video and depth map
        cv2.imshow('Video with Landmarks', frame)
        cv2.imshow('Depth Map', depth_map)

        # Control FPS
        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

        # Exit on 'Esc'
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Save all valid frame data to JSON
if processed_data:
    with open("coordinate_with_depth.json", "w") as f:
        json.dump(processed_data, f, indent=4)
    print("Landmark data with depth saved to coordinate_with_depth.json.")
else:
    print("No valid data found to save.")
