import logging
from main import run_pipeline
from module import load_data

logging.basicConfig(level=logging.DEBUG)

def run_pip_verification():
    run_pipeline(display_mode="json", unlimited=False, save_json=True, mock=True)

    data = load_data("lepaute_data.json")
    print(f"\n--- Verified Local Pipeline Subsystem ({len(data)} Records) ---")
    for idx, item in enumerate(data[:5]):
        print(f"[{idx}] 3D Transform: {[round(x, 4) for x in item['lie_params']]} | ID: {item['detected_object']}")

if __name__ == "__main__":
    run_pip_verification()