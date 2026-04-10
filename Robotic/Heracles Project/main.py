import json
import RPi.GPIO as GPIO
import time

def setup_gpio(motor_pins):
    GPIO.setmode(GPIO.BCM)
    for pin in motor_pins.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

def move_motor(pin, direction, duration):
    """
        pin (int): GPIO pin controlling the motor.
        direction (str): 'positive' or 'negative' movement.
        duration (float): Duration to keep the motor running (in seconds).
    """
    print(f"Moving motor on pin {pin} direction: {direction}, duration: {duration}s")
    GPIO.output(pin, GPIO.HIGH if direction == 'positive' else GPIO.LOW)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)

def load_json(file_path):
    """
        file_path (str): Path to the JSON file.
    Returns:
        list: Loaded data as a list of frames with landmarks.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def calculate_differences(data):
    """
        data (list): List of frames with landmark data.
    Returns:
        list: Frame differences relative to the first frame.
    """
    zero_frame = data[0]["landmarks"]
    differences = []

    for frame in data:
        frame_diff = {}
        for key, coords in frame["landmarks"].items():
            zero_coords = zero_frame[key]
            diff = [coord - zero_coord for coord, zero_coord in zip(coords, zero_coords)]
            frame_diff[key] = diff
        differences.append(frame_diff)
    return differences

def main():
    motor_pins = {
    }

    setup_gpio(motor_pins)

    json_file = "coordinate_with_depth.json"
    data = load_json(json_file)

    differences = calculate_differences(data)

    for frame_index, frame_diff in enumerate(differences):
        print(f"Processing frame {frame_index}")
        for landmark, diff in frame_diff.items():
            if landmark in motor_pins:
                x_diff, y_diff, z_diff = diff
                direction = 'positive' if x_diff > 0 else 'negative'
                duration = abs(x_diff) / 1000.0
                move_motor(motor_pins[landmark], direction, duration)

    GPIO.cleanup()

if __name__ == "__main__":
    main()