import numpy as np
import json
import mediapipe as mp
import open3d as o3d
import time

# Initialize MediaPipe utilities
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Load JSON data
json_path = input("Enter the JSON file path: ")
try:
    with open(json_path, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

# Verify JSON file content
if not isinstance(data, list):
    print("Error: JSON data is not in the expected list format.")
    exit()

# Define pose and hand connections
pose_connections = mp_pose.POSE_CONNECTIONS
hand_connections = mp_holistic.HAND_CONNECTIONS

# Helper function: Create Open3D point cloud from landmarks
def create_point_cloud(landmarks):
    if not landmarks:
        return o3d.geometry.PointCloud()
    points = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks], dtype=np.float64)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# Helper function: Create Open3D line set for connections
def create_line_set(landmarks, connections):
    if not landmarks or not connections:
        return o3d.geometry.LineSet()
    lines = []
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):  # Ensure valid indices
            lines.append([start_idx, end_idx])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(landmarks, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

# Set up Open3D visualizer (single window)
vis = o3d.visualization.Visualizer()
vis.create_window()
target_fps = 30
frame_duration = 1.0 / target_fps  # Target duration per frame

# Prepare placeholders for point cloud and line set
point_cloud = o3d.geometry.PointCloud()
line_set = o3d.geometry.LineSet()
vis.add_geometry(point_cloud)
vis.add_geometry(line_set)

# Playback loop
for frame_idx, frame in enumerate(data):
    start_time = time.time()  # Record the start time of this frame
    landmarks = []
    connections = []

    # Combine pose and hand landmarks into one 3D list
    if "pose" in frame.get("landmarks", {}):
        pose_landmarks = [coord for coord in frame["landmarks"]["pose"].values()]
        landmarks.extend(pose_landmarks)
        connections.extend(pose_connections)

    if "right_hand" in frame.get("landmarks", {}):
        right_hand_landmarks = [coord for coord in frame["landmarks"]["right_hand"].values()]
        landmarks.extend(right_hand_landmarks)
        connections.extend(hand_connections)

    if "left_hand" in frame.get("landmarks", {}):
        left_hand_landmarks = [coord for coord in frame["landmarks"]["left_hand"].values()]
        landmarks.extend(left_hand_landmarks)
        connections.extend(hand_connections)

    # Skip empty frames
    if not landmarks:
        print(f"Skipping frame {frame_idx + 1}: No landmarks found.")
        continue

    # Update the point cloud and line set
    point_cloud.points = o3d.utility.Vector3dVector(np.array(landmarks, dtype=np.float64))
    line_set.points = o3d.utility.Vector3dVector(np.array(landmarks, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector([[start, end] for start, end in connections])

    # Update the Open3D visualizer
    vis.update_geometry(point_cloud)
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()

    # Enforce the playback frame duration
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)

print("Playback completed!")

# Close Open3D window
vis.destroy_window()
