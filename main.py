import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from scipy.spatial import distance
from scipy.interpolate import interp1d
import numpy as np
from collections import deque
# Load the YOLO model (using YOLOv8s for faster inference)
model = YOLO("yolov8s.pt")

def classify_team_by_color(frame, box):
    # Crop the player's bounding box area
    x1, y1, x2, y2 = box
    player_crop = frame[y1:y2, x1:x2]

    # Convert the cropped area to HSV color space
    hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

    # Define color ranges for two teams (example colors)
    team1_lower = np.array([30, 50, 50])  # Adjust based on team colors
    team1_upper = np.array([80, 255, 255])

    team2_lower = np.array([100, 50, 50])
    team2_upper = np.array([140, 255, 255])

    # Calculate the color mask for each team
    team1_mask = cv2.inRange(hsv_crop, team1_lower, team1_upper)
    team2_mask = cv2.inRange(hsv_crop, team2_lower, team2_upper)

    # Determine team based on mask area
    if cv2.countNonZero(team1_mask) > cv2.countNonZero(team2_mask):
        return 'team1'  # Green box
    else:
        return 'team2'  # Red box
def detect_objects(frame):
    # Run YOLO model on the frame
    results = model(frame)

    # Assuming `results` is a list where results[0] contains the primary detection info
    detections = results[0].boxes  # Updated access method
    players = []
    ball = None
    player_class_id = 0  # Class ID for "person" in COCO dataset

# Placeholder for ball ID - adjust if a custom-trained model with a ball class is available
    ball_class_id = 32  # Assuming '32' is for ball in a custom-trained YOLO model

    # Iterate over detected objects
    for detection in detections:
        cls = int(detection.cls[0])  # Class ID of the detected object
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates

        if cls == player_class_id:  # Assuming `player_class_id` is defined for players
            players.append((x1, y1, x2, y2))
        elif cls == ball_class_id:  # Assuming `ball_class_id` is defined for the ball
            ball = (x1, y1, x2, y2)

    return players, ball

def track_ball_and_players(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    ball_positions = deque()  # Store ball positions for interpolation

    if not cap.isOpened():
        print("Error: Could not open video.")
        return frames

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or can't read frame.")
            break

        frame_count += 1
        players, ball = detect_objects(frame)

        # Handle ball position tracking with interpolation
        if ball:
            ball_positions.append((frame_count, ball))
            cv2.rectangle(frame, (ball[0], ball[1]), (ball[2], ball[3]), (0, 0, 255), 2)

        # Draw player bounding boxes with team color classification
        for (x1, y1, x2, y2) in players:
            team = classify_team_by_color(frame, (x1, y1, x2, y2))
            if team == 'team1':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

        frames.append(frame)

    # Interpolate ball positions if missing
    if len(ball_positions) > 1:
        frame_indices, positions = zip(*ball_positions)
        x_coords = [(pos[0] + pos[2]) // 2 for pos in positions]  # Center x of ball
        y_coords = [(pos[1] + pos[3]) // 2 for pos in positions]  # Center y of ball

        interp_x = interp1d(frame_indices, x_coords, kind='linear', fill_value='extrapolate')
        interp_y = interp1d(frame_indices, y_coords, kind='linear', fill_value='extrapolate')

        for i, (frame_idx, ball) in enumerate(ball_positions):
            if ball is None:
                est_x = int(interp_x(frame_idx))
                est_y = int(interp_y(frame_idx))
                ball_positions[i] = (frame_idx, (est_x - 5, est_y - 5, est_x + 5, est_y + 5))

    cap.release()
    print(f"Total frames processed: {frame_count}")
    return frames
def calculate_speed(prev_position, curr_position, frame_rate):
    if prev_position is None or curr_position is None:
        return 0
    dist_pixels = distance.euclidean(prev_position, curr_position)
    # Convert pixels to meters if known, otherwise assume scaling factor
    speed = dist_pixels * frame_rate
    return speed
def calculate_angle(start_pos, mid_pos, end_pos):
    if start_pos and mid_pos and end_pos:
        a = np.array(start_pos)
        b = np.array(mid_pos)
        c = np.array(end_pos)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    return None
def calculate_distance(start_pos, end_pos):
    return distance.euclidean(start_pos, end_pos) if start_pos and end_pos else 0
def display_info(frame, speed, angle, distance):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (50, 50), font, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 80), font, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Distance: {distance:.2f} m", (50, 110), font, 0.5, (255, 255, 255), 2)

def main():
    # Define the path to your video file
    video_path = r"C:\Users\ayush\OneDrive\Football_Analysis\data\8K _ Messi Free Kick Goal vs Liverpool.mp4"

    
    # Process the video and get frames with tracked players and ball
    processed_frames = track_ball_and_players(video_path)
    
    # Save or display each processed frame
    output_path = "outputs/processed_video.avi"
    save_processed_video(processed_frames, output_path)

    print("Video processing complete! The output is saved in:", output_path)

def save_processed_video(frames, output_path, fps=30):
    # Assuming frames are of the same size, get the dimensions from the first frame
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Processed video saved at {output_path}")

# Call the main function when the script is executed
if __name__ == "__main__":
    main()
