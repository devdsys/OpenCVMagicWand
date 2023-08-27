import cv2 as cv
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter  # For TFLite model
import RPi.GPIO as GPIO

# Set up Raspberry Pi pins
GPIO.setmode(GPIO.BCM)
GPIO_RELAY1 = 17  # physical pin 11
GPIO_RELAY2 = 27  # physical pin 13
GPIO.setup(GPIO_RELAY1, GPIO.OUT)
GPIO.setup(GPIO_RELAY2, GPIO.OUT)
GPIO.output(GPIO_RELAY1, True)
GPIO.output(GPIO_RELAY2, True)

# Initialize state variable for falling edge detection
previous_state = 0

# Flag for creating a blank image once, based on the image size
start_state = 1

# Define the classes for the TFLite model
classes = ["no_class", "offAll", "offLeft", "offRight",
           "onAll", "onLeft", "onRight"]  # Class names (order is important!)

# Load the TFLite model
model_path = 'base_model.tflite'
interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

# Allocate tensors for the interpreter
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the inference function
def inference(img):
    # Resize the image to the required dimensions and convert to tensor
    img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
    input_tensor = np.array(np.expand_dims(img, 0), dtype=np.float32)
    input_tensor = np.array(np.expand_dims(input_tensor, -1), dtype=np.float32)

    # Invoke the interpreter to make a prediction
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    res = np.argmax(pred)

    # Map the prediction to a class and perform corresponding GPIO operations
    print(f"Class: {classes[res]}")
    if classes[res] == "onLeft":
        GPIO.output(GPIO_RELAY1, False)
    elif classes[res] == "onRight":
        GPIO.output(GPIO_RELAY2, False)
    elif classes[res] == "onAll":
        GPIO.output(GPIO_RELAY1, False)
        GPIO.output(GPIO_RELAY2, False)
    elif classes[res] == "offLeft":
        GPIO.output(GPIO_RELAY1, True)
    elif classes[res] == "offRight":
        GPIO.output(GPIO_RELAY2, True)
    elif classes[res] == "offAll":
        GPIO.output(GPIO_RELAY1, True)
        GPIO.output(GPIO_RELAY2, True)
    else:
        print("***Incorrect class***")


# Initialize the webcam for image capture
cam = cv.VideoCapture(0)

# Initialize an empty list to store the points from the contours
points_list = []

while True:
    # Read an image frame from the webcam
    t, img = cam.read()

    # Flip the image and apply a Gaussian blur to smooth the image and remove noise
    # The resulting image will appear as a mirror view
    img = cv.flip(img, 1)
    img = cv.GaussianBlur(img, (17, 17), 0)

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to the grayscale image to isolate large spots of light
    ret, thresh1 = cv.threshold(gray, 210, 255, cv.THRESH_BINARY)
    opening = cv.morphologyEx(thresh1, cv.MORPH_OPEN, (7, 7))

    # Find contours of light spots in the processed image
    cnts, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a blank image of the same size as the original (only once)
    if start_state:
        blank = np.ones(img.shape, dtype=np.uint8)
        start_state = 0

    # If there were contours in the previous state but not in the current state (falling edge of signal), make a prediction
    if previous_state and not cnts:
        # Draw a line between all collected points on the blank image
        for d in range(len(points_list) - 1):
            cv.line(blank, points_list[d], points_list[d + 1], (255, 255, 255), 10)
        # Convert the image to grayscale
        collected_image = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)

        # Make a prediction based on the drawn image
        inference(collected_image)

        # Reset the blank image and previous state
        blank = np.ones(img.shape, dtype=np.uint8)
        previous_state = 0

    # Drawing process:
    # If there are contours in the current state, calculate the center of the largest contour and save the point
    if cnts:
        previous_state = 1
        c = max(cnts, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        center = (int(x + (w / 2)), int(y + (h / 2)))
        points_list.append(center)
        cv.circle(gray, center, 10, (0, 0, 255), -1)

        # Draw the line between adjacent contour points based on the current and previous points
        if len(points_list) > 1:
            for d in range(len(points_list) - 1):
                # Display the image which the user can see
                cv.line(img, points_list[d], points_list[d + 1], (0, 0, 255), 10)
    else:
        points_list = []

    # Display the image with contours (if any)
    cv.imshow("Result", img)

    # Wait for a key press
    key = cv.waitKey(30) & 0xFF
    # If the `q` key was pressed, break from the loop
    if key == ord('q'):
        cv.destroyWindow('Result')
        break