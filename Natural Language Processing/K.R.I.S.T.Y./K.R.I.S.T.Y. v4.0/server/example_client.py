import requests
import base64
import json
import numpy as np

API_URL = "http://127.0.0.1:8000/api/inference"
HEALTH_URL = "http://127.0.0.1:8000/api/health"

def check_health():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        print(f"[+] Server Health: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[-] Server unreachable: {e}")
        return False

def query_kristy(prompt: str):
    payload = {"message": prompt}
    headers = {"Content-Type": "application/json"}

    print(f"[*] Dispatching query: '{prompt}'")
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[-] API Error {response.status_code}: {response.text}")
            return

        data = response.json()
        
        print("\n=== K.R.I.S.T.Y. TEXT RESPONSE ===")
        print(data.get("text_response", "No text generated."))
        
        telemetry = data.get("telemetry", {})
        print(f"\n[Telemetry] Processing Time: {telemetry.get('generation_time_sec', 0):.2f}s")
        print(f"[Telemetry] Intent: {data.get('intent')} | Alignment: {data.get('alignment_score', 0):.3f}")

        motion_raw = data.get("motion_data", [])
        if motion_raw:
            motion_matrix = np.array(motion_raw)
            print(f"\n[+] Motion Matrix received. Shape: {motion_matrix.shape}")
            print(f"    Total Frames: {motion_matrix.shape[0]}")
            print(f"    Degrees of Freedom: {motion_matrix.shape[1] // 3}")
            root_joint_frame_0 = motion_matrix[0][:3]
            print(f"    Frame 0 Root Joint [X, Y, Conf]: {root_joint_frame_0}")
        else:
            print("\n[-] No motion data returned.")

        b64_img = data.get("image_base64", "")
        if b64_img:
            output_filename = "kristy_output_latent.jpg"
            try:
                img_bytes = base64.b64decode(b64_img)
                with open(output_filename, "wb") as f:
                    f.write(img_bytes)
                print(f"\n[+] Image successfully decoded and saved to '{output_filename}'")
            except Exception as e:
                print(f"\n[-] Failed to decode image: {e}")

    except requests.exceptions.RequestException as e:
        print(f"[-] Request failed: {e}")

if __name__ == "__main__":
    if check_health():
        print("-" * 50)
        test_prompt = "Initialize forward pass sequence for 5-segment rigid shell module articulation."
        query_kristy(test_prompt)