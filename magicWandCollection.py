import cv2 as cv
import numpy as np
import os

# Initialize state variable for falling edge detection
previous_state = 0
# Flag for creating a blank image once, based on the size of the image
start_state = 1

# Image saving settings:
# Name of the class, also used as a folder name for saving
name_of_class = 'heart'
# Define the path to the directory for saving the images
save_dir_path = f"/home/damagedsystem/Desktop/POTTER/{name_of_class}"

# Initialize the webcam for image capture
cam = cv.VideoCapture(0)

# Initialize an empty list to store the points from the contours
points_list = []

# Start a continuous loop to capture the images
while True:

    # Read an image frame from the webcam
    t, img = cam.read()

    # Flip the image and apply a Gaussian blur to smooth the image and remove noise
    img = cv.flip(img, 1)  # The obtained image should look like a mirror
    img = cv.GaussianBlur(img, (17, 17), 0)

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to the grayscale image to find large spots of light
    ret, thresh1 = cv.threshold(gray, 210, 255, cv.THRESH_BINARY)
    opening = cv.morphologyEx(thresh1, cv.MORPH_OPEN, (7, 7))

    # Find contours of light spots in the processed image
    cnts, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a blank image of the same size as the original (only once)
    if start_state:
        blank = np.ones(img.shape, dtype=np.uint8)
        start_state = 0

    # Finished image saving process
    # If there were contours in the previous state but not in the current state (falling edge of signal), save the image
    if previous_state and not cnts:
        # Draw a line between all collected points on blank image
        for d in range(len(points_list) - 1):
            cv.line(blank, points_list[d], points_list[d + 1], (255, 255, 255), 10)
        # Convert image to grayscale
        collected_image = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
        # Save image:
        # Count the current number of files in the folder
        cnt = len(next(os.walk(save_dir_path))[2]) + 1
        # Create a folder if it doesn't exist
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        cv.imwrite(f"{save_dir_path}/{save_dir_path.split('/')[-1]}_{cnt}.jpg", collected_image)
        print(f"Class: {name_of_class}. Number of files: {cnt}")

        # Reset the blank image and previous state
        blank = np.ones(img.shape, dtype=np.uint8)
        previous_state = 0

    # Drawing process:
    # If there are contours in the current state, calculate the center of the biggest spot and save the point
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
                # Image which the user can see
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