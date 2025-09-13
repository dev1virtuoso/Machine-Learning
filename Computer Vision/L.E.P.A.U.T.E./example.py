import asyncio
import os
import cv2
import numpy as np
from LEPAUTE import main, get_collected_data
from data_access import load_data
import multiprocessing

async def display_all_data_window():
    """Display all collected data in a single GUI window in real-time."""
    window_name = "All Collected Data"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)  # Green text
    thickness = 1
    line_spacing = 20

    while True:
        data = get_collected_data()
        height = max(100, 40 + len(data) * line_spacing * 8)
        img = np.zeros((height, 800, 3), dtype=np.uint8)  # Black background

        cv2.putText(img, "Collected Data Summary", (10, 20), font, font_scale, color, thickness)

        if len(data) == 0:
            cv2.putText(img, "No data collected yet.", (10, 40), font, font_scale, color, thickness)
        else:
            for i, item in enumerate(data):
                y = 40 + i * line_spacing * 8
                cv2.putText(img, f"Entry {i + 1}:", (10, y), font, font_scale, color, thickness)
                cv2.putText(img, f"  Image1 shape: {len(item['image1'])}, {len(item['image1'][0])}, {len(item['image1'][0][0])}",
                            (10, y + line_spacing), font, font_scale, color, thickness)
                cv2.putText(img, f"  Image2 shape: {len(item['image2'])}, {len(item['image2'][0])}, {len(item['image2'][0][0])}",
                            (10, y + 2 * line_spacing), font, font_scale, color, thickness)
                cv2.putText(img, f"  SO(2) theta: {item['lie_params'][0][0]:.4f}",
                            (10, y + 3 * line_spacing), font, font_scale, color, thickness)
                cv2.putText(img, f"  SE(2) params (tx, ty): ({item['lie_params'][0][1]:.2f}, {item['lie_params'][0][2]:.2f})",
                            (10, y + 4 * line_spacing), font, font_scale, color, thickness)
                cv2.putText(img, f"  Model output: {[f'{x:.2f}' for x in item['output'][0][:3]]}...",
                            (10, y + 5 * line_spacing), font, font_scale, color, thickness)
                cv2.putText(img, f"  Loss: {item['loss']:.4f}",
                            (10, y + 6 * line_spacing), font, font_scale, color, thickness)
                cv2.putText(img, f"  Label: {item['label']}",
                            (10, y + 7 * line_spacing), font, font_scale, color, thickness)

        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
        await asyncio.sleep(0.1)  # Update every 100ms to reduce CPU load

    cv2.destroyWindow(window_name)

async def run_pipeline_and_access_data(display_mode: str = "json", frames_dir: str = "frames", save_json: bool = False, save_image: bool = False):
    print(f"Resource usage: Dynamic adjustment to maintain 10 FPS, targeting ~50% CPU ({multiprocessing.cpu_count()//2} threads initially) and ~50% GPU.")
    print("Monitor usage in Activity Monitor (macOS) under CPU and GPU tabs.")
    print(f"\nStarting the LEPAUTE pipeline in {display_mode} mode...")
    if save_image:
        print(f"Saving frames to {frames_dir}...")
    
    if display_mode in ["gui", "realtime"]:
        asyncio.create_task(display_all_data_window())
    
    await main(display_mode=display_mode, frames_dir=frames_dir, unlimited=True, save_json=save_json, save_image=save_image)
    
    print("\nAccessing collected data from memory...")
    data = get_collected_data()
    
    print(f"Total data entries: {len(data)}")
    if len(data) == 0:
        print("No data collected. Check webcam, feature extraction, or image texture.")
    for i, item in enumerate(data):
        print(f"\nData entry {i + 1}:")
        print(f"  Image1 shape: {len(item['image1'])}, {len(item['image1'][0])}, {len(item['image1'][0][0])}")
        print(f"  Image2 shape: {len(item['image2'])}, {len(item['image2'][0])}, {len(item['image2'][0][0])}")
        print(f"  SO(2) theta: {item['lie_params'][0][0]:.4f}")
        print(f"  SE(2) params (tx, ty): ({item['lie_params'][0][1]:.2f}, {item['lie_params'][0][2]:.2f})")
        print(f"  Model output: {item['output']}")
        print(f"  Loss: {item['loss']}")
        print(f"  Label: {item['label']}")

    if display_mode == "json" and save_json:
        print("\nAccessing data from file...")
        file_data = load_data("lepaute_data.json")
        print(f"Total file data entries: {len(file_data)}")
        if len(file_data) == 0:
            print("No data in file. Ensure pipeline ran successfully in json mode.")

if __name__ == "__main__":
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Choose mode here: 'json', 'gui', or 'realtime'
    mode = "realtime"  # Change to 'json' or 'gui' as needed
    save_json = False  # Set to True to save JSON in json mode
    save_image = False  # Set to True to save frames in json or gui mode
    
    print(f"Running in {mode.upper()} mode...")
    print(f"Save JSON: {save_json}, Save Image: {save_image}")
    asyncio.run(run_pipeline_and_access_data(display_mode=mode, frames_dir=frames_dir, save_json=save_json, save_image=save_image))
