import open3d as o3d
import json
import numpy as np
import time

# Function to load JSON data containing 3D landmark coordinates
def load_json_data(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

# Function to visualize 3D landmarks (pose + hands, excluding face) in Open3D
def visualize_3d_pose_and_hands(json_file):
    # Load the JSON data
    data = load_json_data(json_file)

    # Create an Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose and Hands Visualization")

    # Initialize point cloud object
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)

    print("Starting 3D Playback...")

    # Iterate through each frame in the JSON data
    for frame_data in data:
        landmarks = []
        
        # Extract pose landmarks (body points)
        if "pose" in frame_data["landmarks"]:
            landmarks.extend(list(frame_data["landmarks"]["pose"].values()))
        
        # Extract right hand landmarks
        if "right_hand" in frame_data["landmarks"]:
            landmarks.extend(list(frame_data["landmarks"]["right_hand"].values()))
        
        # Extract left hand landmarks
        if "left_hand" in frame_data["landmarks"]:
            landmarks.extend(list(frame_data["landmarks"]["left_hand"].values()))

        # Convert landmarks to a NumPy array
        points = np.array(landmarks, dtype=np.float64)

        # If no landmarks are present, skip the frame
        if points.size == 0:
            print(f"Skipping empty frame {frame_data['frame']}")
            continue

        # Update the point cloud data
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Update the visualization
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # Add a small delay to simulate frame-by-frame playback
        time.sleep(0.1)  # Adjust playback speed by changing the delay

    print("3D Playback Complete.")
    vis.destroy_window()

# Path to the JSON file containing 3D pose and hand landmarks
json_file_path = "coordinate_with_depth.json"

# Visualize the JSON data in a 3D window
visualize_3d_pose_and_hands(json_file_path)
