import cv2
import numpy as np

def preprocess_image(image):
    resized_image = cv2.resize(image, (640, 480))

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    equalized_image = cv2.equalizeHist(gray_image)

    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    edges = cv2.Canny(blurred_image, 100, 200)

    return edges

image = cv2.imread('input_image.jpg')

preprocessed_image = preprocess_image(image)

cv2.imshow('Preprocessed Image', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
