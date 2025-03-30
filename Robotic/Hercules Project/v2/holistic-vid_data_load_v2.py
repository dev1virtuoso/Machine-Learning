import cv2
import numpy as np
import json
import mediapipe as mp
import time

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Load JSON data
json_path = "coordinate.json"
try:
    with open(json_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File '{json_path}' not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_path}'.")
    exit()

# Verify JSON file content
if not isinstance(data, list):
    print("Error: JSON data is not in the expected list format.")
    exit()

# Define output resolution and create black background at 1920x1080
output_width, output_height = 1920, 1080
blank_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

# Define pose and hand landmark names
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

pose_connections = mp_pose.POSE_CONNECTIONS
hand_connections = mp_holistic.HAND_CONNECTIONS

# Function to draw landmarks and connections for a single frame
def draw_landmarks(image, frame_data):
    if "pose" in frame_data["landmarks"]:
        pose_landmarks = frame_data["landmarks"]["pose"]
        for key, coord in pose_landmarks.items():
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circles for pose landmarks
        for connection in pose_connections:
            start_idx, end_idx = connection
            if start_idx < len(pose_tubuh) and end_idx < len(pose_tubuh):  # Safety check for index bounds
                start_key = pose_tubuh[start_idx]
                end_key = pose_tubuh[end_idx]
                if start_key in pose_landmarks and end_key in pose_landmarks:
                    start_point = tuple(map(int, pose_landmarks[start_key]))
                    end_point = tuple(map(int, pose_landmarks[end_key]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # Green lines for pose connections

    if "right_hand" in frame_data["landmarks"]:
        right_hand_landmarks = frame_data["landmarks"]["right_hand"]
        for key, coord in right_hand_landmarks.items():
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Blue circles for right hand landmarks
        for connection in hand_connections:
            start_idx, end_idx = connection
            if start_idx < len(pose_tangan) and end_idx < len(pose_tangan):  # Safety check for index bounds
                start_key = pose_tangan[start_idx]
                end_key = pose_tangan[end_idx]
                if start_key in right_hand_landmarks and end_key in right_hand_landmarks:
                    start_point = tuple(map(int, right_hand_landmarks[start_key]))
                    end_point = tuple(map(int, right_hand_landmarks[end_key]))
                    cv2.line(image, start_point, end_point, (255, 0, 0), 2)  # Blue lines for right hand connections

    if "left_hand" in frame_data["landmarks"]:
        left_hand_landmarks = frame_data["landmarks"]["left_hand"]
        for key, coord in left_hand_landmarks.items():
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red circles for left hand landmarks
        for connection in hand_connections:
            start_idx, end_idx = connection
            if start_idx < len(pose_tangan_2) and end_idx < len(pose_tangan_2):  # Safety check for index bounds
                start_key = pose_tangan_2[start_idx]
                end_key = pose_tangan_2[end_idx]
                if start_key in left_hand_landmarks and end_key in left_hand_landmarks:
                    start_point = tuple(map(int, left_hand_landmarks[start_key]))
                    end_point = tuple(map(int, left_hand_landmarks[end_key]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), 2)  # Red lines for left hand connections

# Process all frames in JSON while controlling FPS
target_fps = 30
frame_duration = 1.0 / target_fps  # Calculate the target duration per frame

for frame in data:
    start_time = time.time()  # Track the start time for each frame

    frame_image = blank_image.copy()  # Start with a fresh black background
    draw_landmarks(frame_image, frame)  # Draw landmarks for the current frame
    cv2.imshow("Frame", frame_image)  # Show the rendered frame

    elapsed_time = time.time() - start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)  # Maintain the target frame duration

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
