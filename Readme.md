# Smart Face Recognition Attendance System (Real-Time using YuNet + FaceNet)

---

## Overview

This project is a **real-time Face Recognition Attendance System** that automatically detects and recognizes faces from **live video feeds** — using either a **webcam** or a **5G Lab IP camera (RTSP)**.

It uses:

* **YuNet** (OpenCV Zoo ONNX Model) for **fast and accurate face detection**
* **FaceNet** (via `keras-facenet`) for **face embedding generation**
* **TensorFlow** for deep feature extraction
* **OpenCV** for camera integration and visualization
* **Pandas** for attendance management via CSV logging

The system works entirely offline, ensuring privacy and reliability.

---

## Technical Flow

1️⃣ **Face Detection (YuNet)**
→ Input: BGR frame
→ Output: Face bounding boxes + confidence scores

2️⃣ **Embedding Generation (FaceNet)**
→ Input: Cropped face (160x160 RGB)
→ Output: 128D numerical vector

3️⃣ **Face Recognition (Matching)**
→ Euclidean distance between embeddings
→ Match if `distance < 0.8`

4️⃣ **Attendance Logging (Pandas)**
→ Writes entry in `attendance_records/attendance_<date>.csv`

---

## Key Features

* Real-time face detection and recognition using YuNet + FaceNet
* Deep learning embeddings for precise recognition
* Persistent local database (DeepFaceEncodings.pkl)
* Easy student registration using laptop camera
* Automatic daily CSV attendance logging
* Register once → Recognize automatically every day

---

## Project Structure

```
FaceRecognitionAttendance/
│
├── Images/                          # Stores registered student face images
├── attendance_records/              # Stores daily attendance CSV files
│
├── face_model.py                    # Face detection & embedding module (YuNet + FaceNet)
├── registration.py                  # Register new students using webcam
├── main.py                          # Real-time attendance recognition (IP camera)
│
├── config.json                      # Configuration file for paths and folders
├── DeepFaceEncodings.pkl            # Encoded embeddings database (auto-created)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation (this file)
```

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/<your-username>/FaceRecognitionAttendance.git
cd FaceRecognitionAttendance
```

---

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.\.venv\Scripts\activate
```

**Linux/Mac:**

```bash
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```text
opencv-python==4.10.0.84
numpy==1.26.4
pandas==2.2.2
keras-facenet==0.3.2
tensorflow==2.15.0
protobuf<7,>=3.20
cvzone==1.6.1
```

---

### Download YuNet Model

YuNet is the face detector used in this project.
Download the **YuNet ONNX model** from the official OpenCV Zoo repository:

 **Download Link:**
[face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)

Then place it in the **root directory** of your project:

```
FaceRecognitionAttendance/
│
├── face_detection_yunet_2023mar.onnx
├── face_model.py
└── ...
```

---

### Setup Configuration

Edit `config.json` file to match your folder names:

```json
{
  "images_folder": "Images",
  "encoding_file": "DeepFaceEncodings.pkl",
  "attendance_folder": "attendance_records"
}
```

---

## System Components

### `face_model.py`

* Loads **YuNet** ONNX model for detection
* Loads **FaceNet** for embedding extraction
* Functions:

  * `detect_single_face(frame)` → Detects one face
  * `get_faces_and_embeddings(frame)` → Detects multiple faces & gets embeddings
  * `get_embedding_from_crop(face_crop)` → Returns embedding for cropped face

---

### `registration.py`

* Uses **laptop webcam** for face registration
* Detects faces using YuNet
* Press **‘s’** to save face and embedding
* Stores all data locally in:

  * `/Images/` (face images)
  * `DeepFaceEncodings.pkl` (embeddings + IDs + names)

---

### `main.py`

* Connects to **RTSP stream** or webcam feed
* Detects multiple faces per frame
* Generates embeddings for each detected face
* Matches embeddings with known database using Euclidean distance
* Marks attendance once per person per day

---


## Step-by-Step Usage

### Step 1: Register New Students

Run:

```bash
python registration.py
```

**Instructions:**

1. Enter **student name** and **roll number**.
2. Look straight into the webcam (green box will appear).
3. Press **‘s’** to save your registration.
4. To register more faces, continue entering details.
5. Press **‘q’** to exit.

**Output:**

* A cropped face image saved in `/Images`
* A 128D embedding stored in `DeepFaceEncodings.pkl`

---

### Step 2: Run Real-Time Attendance (Using Sparsh 5G IP Camera)

This system supports any IP camera that provides an **RTSP stream** — here we are using **Sparsh 5G CCTV cameras**.

---

#### What is RTSP?

**RTSP (Real-Time Streaming Protocol)** is a standard network protocol used by IP cameras to transmit **live video feeds** over a local network.
Each camera generates a **unique RTSP URL**, which can be used in software (like OpenCV) to fetch real-time frames.

**Example RTSP URL format:**

```
rtsp://<username>:<password>@<camera_ip>:<port>/<path>
```

For **Sparsh 5G CCTV cameras**, the most common URL pattern is:

```
rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1-substream.sdp
```

Where:

| Component                                    | Description                                                 |
| -------------------------------------------- | ----------------------------------------------------------- |
| `admin`                                      | Camera login username                                       |
| `admin123`                                   | Camera password                                             |
| `192.168.128.10`                             | Local IP address of the camera (change based on your setup) |
| `554`                                        | Default RTSP port (used by most cameras)                    |
| `/avstream/channel=1/stream=1-substream.sdp` | Camera feed path (substream for lower resolution)           |

You can verify the RTSP URL by opening it in **VLC Media Player** →
`Media > Open Network Stream > Paste RTSP URL`.

If the feed opens in VLC, it will also work with OpenCV.

---

#### Configure in `main.py`

Edit your credentials in the code:

```python
username = "admin"
password = "admin123"
ip_camera_url = f"rtsp://{username}:{password}@192.168.128.10:554/avstream/channel=1/stream=1-substream.sdp"
```

Then run:

```bash
python main.py
```

 The program will:

* Connect to your Sparsh 5G camera’s live feed
* Detect and recognize student faces in real-time
* Mark attendance automatically once per day
* Save attendance logs in `/attendance_records/`

---

### Output Files

**Attendance CSV Example:**

```
attendance_records/attendance_2025-11-10.csv
```

| rollno   | name                 | datetime            |
| -------- | -------------------- | ------------------- |
| u22ec145 | Abhisar Kumar        | 2025-11-10 16:45:12 |
| u22ec166 | Jyotishankar Sahoo   | 2025-11-10 16:47:33 |

---

## Troubleshooting

**No face detected?**

* Ensure `face_detection_yunet_2023mar.onnx` exists in root folder.
* Increase lighting or reduce threshold (in `face_model.py` → `confidence < 0.7`).

**Faces misidentified?**

* Lower recognition threshold to `0.75` in `main.py`.

**CSV not updating?**

* Check if `attendance_records` folder exists (auto-created on first run).
* Ensure `DeepFaceEncodings.pkl` contains valid embeddings.

---

## Future Enhancements
🔹 Streamlit-based web dashboard
🔹 Auto-tracking of multiple persons via DeepSORT
🔹 Cloud-sync of attendance logs

---

## 📜 License

This project is open-source and licensed under the **MIT License**.
You are free to modify and use it for research, educational, or development purposes.
