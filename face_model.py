import os
import cv2
import numpy as np
from keras_facenet import FaceNet

# --- ENVIRONMENT SETUP ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- LOAD MODELS ---
# 1. Load FaceNet (For Recognition)
embedder = FaceNet()
print("FaceNet model loaded.")

# 2. Load YuNet (For Detection - FAST)
yunet_path = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(yunet_path):
    print(f"ERROR: '{yunet_path}' not found. Please download it.")
    exit()

# Initialize YuNet
face_detector = cv2.FaceDetectorYN.create(
    model=yunet_path,
    config="",
    input_size=(320, 320),
    score_threshold=0.8, 
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)
print("YuNet detection model loaded.")

def detect_single_face(frame):
    """
    Used by registration.py to find the largest face.
    """
    h, w, _ = frame.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    
    if faces is None:
        return None
        
    # Find the largest face
    best_face = None
    max_area = 0
    for face in faces:
        box = face[0:4].astype(int)
        x, y, w_box, h_box = box
        area = w_box * h_box
        if area > max_area:
            max_area = area
            best_face = (x, y, w_box, h_box)
    return best_face

def get_embedding_from_crop(face_crop):
    """
    Used by registration.py to get embedding from a cropped face.
    """
    face_crop = cv2.resize(face_crop, (160, 160))
    face_pixels = np.expand_dims(face_crop, axis=0)
    embedding = embedder.embeddings(face_pixels)[0]
    return embedding

def get_faces_and_embeddings(frame):
    """
    Used by main.py to detect multiple faces.
    """
    detections = []
    h, w, _ = frame.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    
    if faces is None:
        return []

    for face in faces:
        box = face[0:4].astype(int)
        x, y, w_box, h_box = box
        x, y = max(0, x), max(0, y)
        
        face_crop = frame[y:y+h_box, x:x+w_box]
        if face_crop.size == 0: continue
            
        try:
            embedding = get_embedding_from_crop(face_crop)
            detections.append({
                'box': [x, y, w_box, h_box],
                'embedding': embedding,
                'score': face[-1]
            })
        except:
            continue
            
    return detections
