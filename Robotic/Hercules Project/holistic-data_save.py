import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import json

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Variables for tracking
count = 0
alldata = []
fps_time = time.time()  # Initialize FPS time tracking

# Define body and hand landmarks for mapping
pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

pose_tangan_2 = [name + '2' for name in pose_tangan]  # Automate left hand naming

# Start video capture
cap = cv2.VideoCapture(0)

# Function to save landmark data periodically
def save_landmark_data():
    while True:
        global alldata
        if alldata:
            with open("coordinate.json", "w") as f:
                json.dump(alldata, f)
            print("Data saved to coordinate.json")
            alldata = []  # Clear data after saving
        time.sleep(1)  # Save data every second

# Start data saving thread
threading.Thread(target=save_landmark_data, daemon=True).start()

# Process video feed using MediaPipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Check camera connection.")
            break

        # Flip and convert frame for processing
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False  # Mark image as non-writeable for faster processing
        results = holistic.process(frame)  # Process frame using holistic model

        # Convert back to BGR for visualization
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Create a blank image for landmarks visualization
        blank_image = np.zeros(frame.shape, dtype=np.uint8)

        # Draw landmarks for body, hands, and pose
        mp_drawing.draw_landmarks(blank_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(blank_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(blank_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Collect landmark data if available
        if results.pose_landmarks:
            alldata.append({pose_tubuh[i]: (lm.x * blank_image.shape[1], lm.y * blank_image.shape[0])
                            for i, lm in enumerate(results.pose_landmarks.landmark)})

        if results.right_hand_landmarks:
            alldata.append({pose_tangan[i]: (lm.x * blank_image.shape[1], lm.y * blank_image.shape[0])
                            for i, lm in enumerate(results.right_hand_landmarks.landmark)})

        if results.left_hand_landmarks:
            alldata.append({pose_tangan_2[i]: (lm.x * blank_image.shape[1], lm.y * blank_image.shape[0])
                            for i, lm in enumerate(results.left_hand_landmarks.landmark)})

        # Display FPS on the frame
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()  # Update FPS time
        cv2.putText(blank_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the original and processed frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Processed Landmarks', blank_image)

        # Increment frame count for tracking
        count += 1
        print(f"Frame count: {count}")

        # Exit if 'Esc' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
