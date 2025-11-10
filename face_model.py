# face_model.py
import cv2
import numpy as np
from keras_facenet import FaceNet

# --- 1. Load the Embedder (FaceNet) ---
# This is what creates the 128D vector
print("Loading FaceNet embedder...")
embedder = FaceNet()
print("FaceNet embedder loaded.")

# --- 2. Load the Detector (YuNet) ---
# This is what finds the face in the image
print("Loading YuNet face detector...")
# Download the model file here if you don't have it:
# https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
model_path = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0))
print("YuNet detector loaded.")

def get_embedding_from_crop(face_crop):
    """
    Takes a pre-cropped face (160x160) and returns the embedding.
    """
    face_pixels = np.expand_dims(face_crop, axis=0)
    embedding = embedder.embeddings(face_pixels)
    return embedding[0]

def detect_single_face(frame):
    """
    Detects just one face using fast YuNet.
    Returns a single box [x, y, w, h] or None.
    """
    # YuNet needs a specific input size
    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    # faces[1] contains the list of detected faces
    faces = detector.detect(frame)
    if faces[1] is None:
        return None

    # Find the face with the highest confidence
    best_face = max(faces[1], key=lambda x: x[14]) # index 14 is confidence
    confidence = best_face[14]
    
    # --- MODIFIED --- Lower threshold for bad light
    if confidence < 0.8:
        return None

    # Convert box from [x, y, w, h]
    box = best_face[:4].astype(int)
    return box

def get_faces_and_embeddings(frame):
    """
    Takes a whole BGR frame, finds faces (YuNet),
    and returns a list of boxes and embeddings (FaceNet).
    """
    detections = []
    
    # YuNet needs a specific input size
    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    faces = detector.detect(frame)
    if faces[1] is None:
        return detections # Return empty list

    # Loop over all detected faces
    for face_data in faces[1]:
        confidence = face_data[14]
        
        # --- MODIFIED --- Lower threshold
        if confidence < 0.85:
            continue
            
        box = face_data[:4].astype(int)
        (x, y, w, h) = box
        
        # Crop the face
        if y < 0: y = 0 # Fix negative cropping
        if x < 0: x = 0
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            continue
            
        # Resize for FaceNet
        face_crop_resized = cv2.resize(face_crop, (160, 160))
        
        # Get embedding
        embedding = get_embedding_from_crop(face_crop_resized)
        
        detections.append({
            'box': box,
            'embedding': embedding
        })
        
    return detections