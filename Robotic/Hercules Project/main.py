import json
import RPi.GPIO as GPIO
import time

# Setup GPIO pins for motors
def setup_gpio(motor_pins):
    GPIO.setmode(GPIO.BCM)  # Use Broadcom SOC pin numbering
    for pin in motor_pins.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)  # Initialize motors to OFF

def move_motor(pin, direction, duration):
    """
    Controls a motor connected to a GPIO pin.
    Args:
        pin (int): GPIO pin controlling the motor.
        direction (str): 'positive' or 'negative' movement.
        duration (float): Duration to keep the motor running (in seconds).
    """
    print(f"Moving motor on pin {pin} direction: {direction}, duration: {duration}s")
    GPIO.output(pin, GPIO.HIGH if direction == 'positive' else GPIO.LOW)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)  # Stop motor

def load_json(file_path):
    """
    Loads the JSON file containing landmark data.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        list: Loaded data as a list of frames with landmarks.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def calculate_differences(data):
    """
    Calculates frame-by-frame differences relative to the first frame.
    Args:
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

# Main program to control motors
def main():
    # GPIO motor pin mapping for landmarks
    motor_pins = {
    }

    # Initialize GPIO pins
    setup_gpio(motor_pins)

    # Load the JSON file with landmarks
    json_file = "coordinate_with_depth.json"  # Replace with your JSON file path
    data = load_json(json_file)

    # Calculate differences between frames and the first frame
    differences = calculate_differences(data)

    # Control motors based on differences
    for frame_index, frame_diff in enumerate(differences):
        print(f"Processing frame {frame_index}")
        for landmark, diff in frame_diff.items():
            if landmark in motor_pins:  # Check if the landmark has a motor pin
                x_diff, y_diff, z_diff = diff  # Unpack XYZ differences
                direction = 'positive' if x_diff > 0 else 'negative'
                duration = abs(x_diff) / 1000.0  # Convert the difference to duration (adjust scaling)
                move_motor(motor_pins[landmark], direction, duration)

    # Cleanup GPIO
    GPIO.cleanup()

if __name__ == "__main__":
    main()