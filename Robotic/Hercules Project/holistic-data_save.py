import cv2
import mediapipe as mp
import time
import os
import threading

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Initialize variables to store landmark data
landmark_data = []
lock = threading.Lock()

# Function to get current time in microseconds
def current_time_microseconds():
    return int(time.time() * 1_000_000)

# Function to save landmark data to a text file
def save_landmark_data():
    while True:
        with lock:
            if landmark_data:
                # Get the current directory path
                current_directory = os.getcwd()
                file_path = os.path.join(current_directory, 'landmark_data.txt')
                try:
                    with open(file_path, 'w') as file:
                        for data in landmark_data:
                            file.write(f"{data[0]}, {data[1]}, {data[2]}, {data[3]}\n")
                    print(f"Landmark data saved to: {file_path}")
                except Exception as e:
                    print(f"An error occurred while saving the file: {e}")
            time.sleep(1)  # Save data every 1 second

# Start the threading for saving landmark data
threading.Thread(target=save_landmark_data, daemon=True).start()

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.9) as holistic:
    
    last_record_time = current_time_microseconds()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.pose_landmarks:
            current_time = current_time_microseconds()
            if current_time - last_record_time >= 100:
                landmarks = results.pose_landmarks.landmark
                last_record_time = current_time

                with lock:
                    for landmark in landmarks:
                        landmark_data.append((current_time, landmark.x, landmark.y, landmark.z))

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
