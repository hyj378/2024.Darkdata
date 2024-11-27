### 비디오에서 프레임을 추출하는 코드입니다.
import cv2
import os

# Set the video path and output directory for frames
video_path = '../videos/parking_lot.avi'
output_dir = '../data/JB_data/frames/frames_the_entrance'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Original FPS of the video

# Calculate the interval to get under 10 frames per second
if fps>10:
    frame_interval = 2
else:
    frame_interval = 1

frame_count = 0
saved_frame_count = 0

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Save under 10 frames per second
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_dir, f'frame_{saved_frame_count:06d}.png')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_frame_count} frames and saved them in '{output_dir}'")