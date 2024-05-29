import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect mouse movement between two frames
def detect_mouse_movement(prev_frame, curr_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between frames
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store mouse movement coordinates
    mouse_movements = []

    for contour in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(contour) > 100:
            # Get the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                mouse_movements.append((cx, cy))

    return mouse_movements

# Function to calculate velocity on the Y-axis
def calculate_velocity_y(prev_coord, curr_coord, time_interval):
    dy = curr_coord[1] - prev_coord[1]
    velocity_y = dy / time_interval
    return velocity_y

# Function to process video and extract velocities and time intervals
def process_video(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Read the first frame
    ret, prev_frame = cap.read()

    # Initialize variables for previous time and mouse position
    prev_time = 0
    prev_mouse_pos = (0, 0)

    # Initialize lists to store velocities and time intervals
    velocities = []
    time_intervals = []

    while(cap.isOpened()):
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Detect mouse movement between frames
        mouse_movements = detect_mouse_movement(prev_frame, curr_frame)

        # Calculate time interval
        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        time_interval = curr_time - prev_time

        # Calculate mouse velocity on the Y-axis
        if len(mouse_movements) > 1:
            curr_mouse_pos = mouse_movements[-1]
            velocity_y = calculate_velocity_y(prev_mouse_pos, curr_mouse_pos, time_interval)
            prev_mouse_pos = curr_mouse_pos
        else:
            velocity_y = 0

        # Store velocity and time interval
        velocities.append(velocity_y)
        time_intervals.append(curr_time)

        # Update the previous time
        prev_time = curr_time

        # Update the previous frame
        prev_frame = curr_frame.copy()

    # Release the video capture object
    cap.release()

    return velocities, time_intervals

# Process the first video
video1_velocities, video1_time_intervals = process_video('normalr9.mp4')

# Process the second video
video2_velocities, video2_time_intervals = process_video('cheater.mp4')

# Plot mouse velocity on the Y-axis over time for both videos
plt.plot(video1_time_intervals, video1_velocities, label='Normal')
plt.plot(video2_time_intervals, video2_velocities, label='Cheater')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity on Y-axis (pixels/second)')
plt.title('Mouse Velocity on Y-axis Over Time')
plt.legend()
plt.grid(True)
plt.show()
