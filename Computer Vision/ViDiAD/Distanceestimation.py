import cv2

def calculate_distance(focal_length, known_width, pixel_width):
    distance = (known_width * focal_length) / pixel_width
    return distance

known_width = 0.2
focal_length = 0.3 

image = cv2.imread('image.jpg')
pixel_width = 100

distance = calculate_distance(focal_length, known_width, pixel_width)
print("Estimated distance:", distance, "meters")
