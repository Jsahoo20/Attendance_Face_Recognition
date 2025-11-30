# main.py
import cv2
import numpy as np
import pickle
import json
import os
import time
from datetime import datetime
import pandas as pd

# Import the fast YuNet function
from face_model import get_faces_and_embeddings

# --- 1. CONFIG & DATABASE SETUP ---
config_file = 'config.json'
if not os.path.exists(config_file):
    config = {'encoding_file': 'DeepFaceEncodings.pkl', 'attendance_folder': 'attendance_records'}
    with open(config_file, 'w') as f:
        json.dump(config, f)
else:
    with open(config_file, 'r') as f:
        config = json.load(f)

encoding_file = config['encoding_file']
if not os.path.exists(encoding_file):
    print("ERROR: No encodings found. Please run registration.py first.")
    exit()

with open(encoding_file, 'rb') as f:
    encodeListKnown, studentIds, studentNames = pickle.load(f)
print(f"Loaded {len(studentIds)} registered students.")

# --- 2. CSV SETUP ---
attendance_folder = config.get('attendance_folder', 'attendance_records')
os.makedirs(attendance_folder, exist_ok=True)
today_str = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(attendance_folder, f"attendance_{today_str}.csv")

if os.path.exists(attendance_file):
    attendance_df = pd.read_csv(attendance_file)
else:
    attendance_df = pd.DataFrame(columns=['rollno', 'name', 'datetime'])
    attendance_df.to_csv(attendance_file, index=False)
    
recognized_ids = set(attendance_df['rollno'].astype(str))

# --- 3. CAMERA SETUP ---
username = "admin"    
password = "admin123" 
ip_camera_url = f"rtsp://{username}:{password}@192.168.1.10:554/stream1"

# SELECT SOURCE: 0 for Webcam, or ip_camera_url for 5G Camera
video_source = 0 
# video_source = ip_camera_url 

quit_flag = False

print("\n--- ATTENDANCE SYSTEM STARTED (OPTIMIZED) ---")
print("Press 'q' to Quit.")

while not quit_flag:
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Keep buffer small for real-time feel

    if not cap.isOpened():
        print("Connecting to camera...")
        time.sleep(2)
        continue

    # --- OPTIMIZATION VARIABLES ---
    frame_count = 0
    SKIP_FRAMES = 5   # Run AI only every 5th frame
    SCALE_FACTOR = 0.5 # Shrink image by 50% for processing (Faster)
    
    # Store the last known faces to display during "skipped" frames
    last_processed_results = [] 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()

        # --- LOGIC: PROCESS ONLY EVERY 5th FRAME ---
        if frame_count % SKIP_FRAMES == 0:
            
            # 1. Resize the frame for the AI (Makes it 4x faster)
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            
            # 2. Run AI on small frame
            detections = get_faces_and_embeddings(small_frame)
            
            # 3. Clear old results
            current_results = []
            
            for face in detections:
                # Get coordinates from small frame
                small_box = face['box']
                embedding = face['embedding']
                
                # Scale coordinates back up to normal size
                x = int(small_box[0] * (1 / SCALE_FACTOR))
                y = int(small_box[1] * (1 / SCALE_FACTOR))
                w = int(small_box[2] * (1 / SCALE_FACTOR))
                h = int(small_box[3] * (1 / SCALE_FACTOR))
                
                # Recognition Logic
                distances = [np.linalg.norm(embedding - known) for known in encodeListKnown]
                match_index = np.argmin(distances)
                min_dist = distances[match_index]

                name = "Unknown"
                color = (0, 0, 255) # Red
                
                if min_dist < 0.8:
                    rollno = str(studentIds[match_index])
                    name = studentNames[match_index]
                    color = (0, 255, 0) # Green
                    
                    # Mark Attendance
                    if rollno not in recognized_ids:
                        recognized_ids.add(rollno)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_row = pd.DataFrame([{'rollno': rollno, 'name': name, 'datetime': timestamp}])
                        new_row.to_csv(attendance_file, mode='a', header=False, index=False)
                        print(f"[MARKED] {name}")
                
                # Save this result to display on next 4 frames
                current_results.append((x, y, w, h, name, color))
            
            # Update global results
            last_processed_results = current_results

        # --- DRAWING (Happens every frame so video is smooth) ---
        for (x, y, w, h, name, color) in last_processed_results:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Optimized Attendance System", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
