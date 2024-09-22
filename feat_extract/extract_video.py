import cv2
import os

# Path to the video file and output directory
video_path = '/mnt/experiments/sorlova/datasets/ROL/rgb_video_1000/val/28_M.mp4'
output_dir = './28_M_extracted/'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the original FPS of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Set the desired FPS for extraction
desired_fps = 20
frame_interval = int(original_fps / desired_fps)

frame_count = 0
extracted_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video ends

    # Extract frames at the desired interval
    if frame_count % frame_interval == 0:
        # Save the frame as a JPG image
        output_path = os.path.join(output_dir, f'frame_{extracted_frame_count:04d}.jpg')
        cv2.imwrite(output_path, frame)
        extracted_frame_count += 1

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Extracted {extracted_frame_count} frames at {desired_fps} FPS.")
