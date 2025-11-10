import cv2
import numpy as np
import pickle
import json
import os
import time
from datetime import datetime
import pandas as pd

# --- MODIFIED ---
# This function is now fast and uses YuNet
from face_model import get_faces_and_embeddings

# ... (Config and DB loading... same as before) ...
with open('config.json', 'r') as f:
    config = json.load(f)
encoding_file = config['encoding_file']
if not os.path.exists(encoding_file):
    print("No encodings found. Please register students first.")
    exit()
with open(encoding_file, 'rb') as f:
    encodeListKnown, studentIds, studentNames = pickle.load(f)
print(f"Loaded {len(studentIds)} registered students.")

# ... (CSV loading... same as before) ...
attendance_folder = config.get('attendance_folder', 'attendance_records')
os.makedirs(attendance_folder, exist_ok=True)
today_str = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(attendance_folder, f"attendance_{today_str}.csv")
if os.path.exists(attendance_file):
    attendance_df = pd.read_csv(attendance_file)
else:
    attendance_df = pd.DataFrame(columns=['rollno', 'name', 'datetime'])
    attendance_df.to_csv(attendance_file, index=False)
recognized_ids = set(attendance_df['rollno'])
# ---

# ... (Camera URL and connection... same as before) ...
username = "admin"    # <-- PUT YOUR USERNAME HERE
password = "admin123"  # <-- PUT YOUR PASSWORD HERE
ip_camera_url = f"rtsp://{username}:{password}@192.168.128.10:554/avstream/channel=1/stream=1-substream.sdp"
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
print("Starting attendance system. Press 'q' to quit.")
# ---

# --- MODIFIED --- Removed frame_count
quit_flag = False

while not quit_flag:  # Outer reconnection loop
    cap = None
    try:
        print(f"Attempting to connect to stream...")
        cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            raise IOError("Cannot open video stream")
        
        print("--- Stream connected successfully! ---")

        while True: # Inner processing loop
            ret, frame = cap.read()
            if not ret:
                print("Stream disconnected. Breaking from processing loop.")
                break
            
            display_frame = frame.copy()

            # --- MODIFIED ---
            # Removed frame skipping! We run this every frame now.
            detections = get_faces_and_embeddings(frame)

            # Loop over every face found in the frame
            for face in detections:
                (x, y, w, h) = face['box']
                embedding = face['embedding']
                
                distances = [np.linalg.norm(embedding - known) for known in encodeListKnown]
                match_index = np.argmin(distances)
                min_dist = distances[match_index]

                if min_dist < 0.8: 
                    rollno = studentIds[match_index]
                    name = studentNames[match_index]
                    label = f"{name} ({rollno})"
                    color = (0, 255, 0) # Green

                    if rollno not in recognized_ids:
                        recognized_ids.add(rollno)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_row = {'rollno': rollno, 'name': name, 'datetime': timestamp}
                        attendance_df = pd.concat([attendance_df, pd.DataFrame([new_row])])
                        attendance_df.to_csv(attendance_file, index=False)
                        print(f"Attendance marked for {name} ({rollno}) at {timestamp}")
                else:
                    label = "Unknown"
                    color = (0, 0, 255) # Red

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Smart Face Attendance - IP Camera", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_flag = True
                break 

    except KeyboardInterrupt:
        print("\nStopping on user request (Ctrl+C).")
        quit_flag = True
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if cap is not None:
            cap.release()
        if quit_flag:
            break 
        print("Connection lost. Retrying in 5 seconds...")
        time.sleep(5)

# Cleanup
cv2.destroyAllWindows()
print("Attendance system stopped.")