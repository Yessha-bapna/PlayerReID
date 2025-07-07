# 🏃 Player Re-Identification using YOLOv8 and Deep SORT

This project detects and tracks players in a single video feed using **YOLOv8** for object detection and **Deep SORT** for consistent player re-identification. Each player is assigned a unique ID that persists across video frames, even as they move around or temporarily leave the scene.

The final output is a new video (`tracked_output.mp4`) with bounding boxes and ID labels drawn around each player.

---

## 📂 Project Contents

Ensure the following files are present in your project folder:

- `detect_and_track.py` – Python script for detection and tracking  
- `15sec_input_720p.mp4` – Input soccer video  
- `yolov8n.pt` – YOLOv8 pretrained model (you will learn how to download it)   
- `output/` – Output folder (will be auto-created if missing)

---

## ⚙️ Installation & Setup Instructions

Follow these exact steps to set up and run the project.

---

### ✅ Step 1: Install Python 3.10+

Download Python from the official site:  
👉 https://www.python.org/downloads/

During installation, make sure to check ✅ **"Add Python to PATH"**.

Verify installation by running:

```bash
python --version

### ✅ Step 2: Install Required Libraries
pip install ultralytics==8.0.20
pip install opencv-python
pip install deep_sort_realtime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# These packages include:
- YOLOv8 for detection

- OpenCV for video processing

- Deep SORT for tracking

- PyTorch for model loading

### ✅ Step 3: Download YOLOv8 Model

If you don’t already have yolov8n.pt in your folder, you can download it by running this code in a Python terminal:

from ultralytics import YOLO
YOLO("yolov8n.pt")  # This will download the YOLOv8 nano model

### ✅ Step 4: Run the code

python detect_and_track.py
