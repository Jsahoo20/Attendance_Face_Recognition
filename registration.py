import cv2
import os
import pickle
import numpy as np
from datetime import datetime
import json
import time

# --- MODIFIED ---
# Imports are the same, but the functions are now
# using the new, fast YuNet detector!
from face_model import detect_single_face, get_embedding_from_crop

# ... (Config loading and DB loading... same as before) ...
with open('config.json', 'r') as f:
    config = json.load(f)

images_folder = config['images_folder']
encoding_file = config['encoding_file']

if os.path.exists(encoding_file):
    with open(encoding_file, 'rb') as f:
        encodeListKnown, studentIds, studentNames = pickle.load(f)
else:
    encodeListKnown, studentIds, studentNames = [], [], []

os.makedirs(images_folder, exist_ok=True)
print("Student Registration System\n")
# ... (User prompts... same as before) ...
name = input("Enter student name: ").strip()
rollno = input("Enter roll number: ").strip()
# ---

print("Connecting to laptop camera...")
cap = cv2.VideoCapture(0) # Use laptop camera

if not cap.isOpened():
    print("Error: Could not open laptop camera.")
    exit()

print("--- Camera connected. ---")
print("--- INSTRUCTIONS ---")
print("1. Position your face inside the green box.")
print("2. Press 's' to save the registration.")
print("3. Press 'q' to quit.")


captured = False
quit_flag = False
face_to_save = None

# --- MODIFIED ---
# Removed all frame-skipping logic.
# The new detector is fast enough!
while not captured and not quit_flag:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't read frame from camera.")
            break

        display_frame = frame.copy()
        
        # --- MODIFIED ---
        # This function is now fast. We run it every frame.
        box = detect_single_face(frame)
        
        if box is not None:
            (x, y, w, h) = box
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Found. Press 's' to save.", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Extract the face ROI for saving
            if y < 0: y = 0
            if x < 0: x = 0
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.size > 0:
                face_to_save = cv2.resize(face_crop, (160, 160))
            else:
                face_to_save = None
        
        else:
            cv2.putText(display_frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            face_to_save = None

        cv2.imshow("Registration", display_frame)

        # ... (Key press logic... same as before) ...
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and face_to_save is not None:
            print("Face captured. Generating embedding...")
            
            embedding = get_embedding_from_crop(face_to_save)
            print("Embedding generated. Saving registration...")

            # ... (Saving logic... same as before) ...
            filename = os.path.join(images_folder, f"{rollno}.png")
            cv2.imwrite(filename, face_to_save)
            encodeListKnown.append(embedding)
            studentIds.append(rollno)
            studentNames.append(name)
            with open(encoding_file, 'wb') as f:
                pickle.dump((encodeListKnown, studentIds, studentNames), f)
            # ---

            captured = True
            print(f"Registered student: {name} (Roll No: {rollno})")
            print("You can register another face or press 'q' to quit.")
            
            name = input("Enter student name (or press 'q' to quit): ").strip()
            if name == 'q':
                quit_flag = True
            else:
                rollno = input("Enter roll number: ").strip()
                face_to_save = None
                captured = False

        if key == ord('q'):
            quit_flag = True

    except KeyboardInterrupt:
        print("\nStopping on user request (Ctrl+C).")
        quit_flag = True

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Registration complete.")