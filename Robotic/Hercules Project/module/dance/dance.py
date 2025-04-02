import json
import RPi.GPIO as GPIO
import time

# Initialize GPIO
def setup_gpio(motor_pins):
    GPIO.setmode(GPIO.BCM)  # Use Broadcom SOC numbering
    for pin in motor_pins.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)  # Initialize state to LOW

def move_motor(pin, direction, duration):
    """
    Controls motor movement:
    pin: GPIO pin number for the motor
    direction: Movement direction, 'positive' or 'negative'
    duration: Duration of movement in seconds
    """
    print(f"Moving motor on pin {pin}, direction: {direction}, duration: {duration}s")
    GPIO.output(pin, GPIO.HIGH if direction == 'positive' else GPIO.LOW)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)  # Stop the motor

def load_json_data(json_file):
    """
    Loads JSON data from the given file
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def calculate_differences(data):
    """
    Calculates XYZ differences for each frame relative to the first frame
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

# Main program
def main():
    # GPIO motor mapping: Each landmark corresponds to a GPIO pin
    motor_pins = {
        # Add mappings for additional landmarks and GPIOs
    }
    
    # Set up GPIO
    setup_gpio(motor_pins)
    
    # Load JSON data
    json_file = "coordinate_with_depth.json"
    data = load_json_data(json_file)

    # Calculate differences
    differences = calculate_differences(data)

    # Control motor movements
    for frame_index, frame_diff in enumerate(differences):
        print(f"Processing frame {frame_index}")
        for landmark, diff in frame_diff.items():
            if landmark in motor_pins:  # Ensure the landmark has a corresponding GPIO pin
                x_diff, y_diff, z_diff = diff  # Extract XYZ differences
                direction = 'positive' if x_diff > 0 else 'negative'
                duration = abs(x_diff) / 1000.0  # Convert difference to duration (in seconds)
                move_motor(motor_pins[landmark], direction, duration)

    # Clean up GPIO settings
    GPIO.cleanup()

if __name__ == "__main__":
    main()
