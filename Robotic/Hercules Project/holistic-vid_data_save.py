import cv2
import mediapipe as mp
import numpy as np
import time
import json

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Variables for tracking
target_fps = 30  # Target FPS
frame_duration = 1.0 / target_fps  # Duration per frame (seconds)
processed_data = []  # To store all frames with valid landmark data

# Define body and hand landmarks for mapping
pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

pose_tangan_2 = [name + '2' for name in pose_tangan]  # Dynamic naming for left hand landmarks

# Get video path from user
video_path = input("Enter the MP4 file path: ")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video. Check the file path.")
    exit()

# Function to extract and append landmark data
def extract_landmarks(results, image_shape, frame_count):
    # Initialize data structure for the current frame
    frame_data = {"frame": frame_count, "landmarks": {}}
    has_landmarks = False  # Flag to track if any landmarks exist

    # Pose landmarks
    if results.pose_landmarks:
        frame_data["landmarks"]["pose"] = {
            pose_tubuh[i]: (lm.x * image_shape[1], lm.y * image_shape[0])
            for i, lm in enumerate(results.pose_landmarks.landmark)
        }
        has_landmarks = True

    # Right hand landmarks
    if results.right_hand_landmarks:
        frame_data["landmarks"]["right_hand"] = {
            pose_tangan[i]: (lm.x * image_shape[1], lm.y * image_shape[0])
            for i, lm in enumerate(results.right_hand_landmarks.landmark)
        }
        has_landmarks = True

    # Left hand landmarks
    if results.left_hand_landmarks:
        frame_data["landmarks"]["left_hand"] = {
            pose_tangan_2[i]: (lm.x * image_shape[1], lm.y * image_shape[0])
            for i, lm in enumerate(results.left_hand_landmarks.landmark)
        }
        has_landmarks = True

    # Only add frame data if there are any landmarks
    if has_landmarks:
        processed_data.append(frame_data)

# Process video using MediaPipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_count = 0
    while cap.isOpened():
        start_time = time.time()  # Record frame start time
        success, frame = cap.read()
        if not success:
            print("End of video stream or error reading frame.")
            break

        # Flip and convert frame for processing
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False  # Optimize for processing
        results = holistic.process(frame)

        # Convert back to BGR for visualization
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw landmarks directly on the original frame
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Extract and save landmark data
        frame_count += 1
        extract_landmarks(results, frame.shape, frame_count)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video frames with overlaid landmarks
        cv2.imshow('Video with Landmarks', frame)

        # Control frame duration to maintain target FPS
        if elapsed_time < frame_duration:
            time.sleep(frame_duration - elapsed_time)

        # Exit on pressing 'Esc'
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save all valid frame data to JSON after processing the full video
if processed_data:
    with open("coordinate.json", "w") as f:
        json.dump(processed_data, f, indent=4)  # Pretty-print JSON
    print("All valid frame data has been saved to coordinate.json.")
else:
    print("No valid frame data found. JSON file was not created.")
