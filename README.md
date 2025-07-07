
# 🏃 Player Re-Identification using YOLOv8 and Deep SORT

This project detects and tracks players in a single video feed using **YOLOv8** for object detection and **Deep SORT** for consistent player re-identification. Each player is assigned a unique ID that persists across video frames, even as they move around or temporarily leave the scene.

The final output is a new video (`tracked_output.mp4`) with bounding boxes and ID labels drawn around each player.

---

## 📂 Project Contents

Ensure the following files are present in your project folder:

- `detect_and_track.py` – Python script for detection and tracking  
- `15sec_input_720p.mp4` – Input soccer video  
- `yolov8n.pt` – YOLOv8 pretrained model (you will learn how to download it)  
- *(Optional)* `best.pt` – Custom-trained YOLO model, if available  
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
```

---

### ✅ Step 2: Open the Project Folder in Terminal

Open **Command Prompt** or **PowerShell**, then navigate to your project directory where `detect_and_track.py` and the video are located:

```bash
cd path\to\your\player_reid_single_feed
```

Make sure to replace `path\to\your\...` with the actual path on your system.

---

### ✅ Step 3: Install Required Libraries

Use the following commands to install all required dependencies:

```bash
pip install ultralytics==8.0.20
pip install opencv-python
pip install deep_sort_realtime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

These packages include:
- `ultralytics` for YOLOv8 detection
- `opencv-python` for video handling
- `deep_sort_realtime` for real-time tracking
- `torch`, `torchvision`, `torchaudio` for model inference

> Make sure you're connected to the internet while installing these.

---

### ✅ Step 4: Download the YOLOv8 Model

If you don’t already have the YOLOv8 pretrained model (`yolov8n.pt`), download it using the following Python command:

```python
from ultralytics import YOLO
YOLO("yolov8n.pt")  # Automatically downloads the nano version of YOLOv8
```

Once downloaded, place the `yolov8n.pt` file in your project folder.

You can also use a larger model like `yolov8s.pt`, or a custom one (`best.pt`) if provided.

---

### ✅ Step 5: Run the Script

Now that everything is set up, run the following command:

```bash
python detect_and_track.py
```

What happens:
- The video will open in a new window.
- Each player will be detected and assigned a consistent ID.
- Press `Q` to stop the video preview anytime.
- The output will be saved to:

```bash
output/tracked_output.mp4
```

---

## 🔁 Optional: Using a Custom Model (`best.pt`)

If you have a custom-trained model like `best.pt`, change this line in the script:

```python
model = YOLO("yolov8n.pt")
```

to:

```python
model = YOLO("best.pt")
```

Make sure `best.pt` is in the same directory. If you get an error, the model may not be compatible with the installed YOLO version — ask your mentor for guidance.

---

## 📦 Output

The result will be saved as:

```
output/tracked_output.mp4
```

This output shows each player with a bounding box and a unique, consistent ID across frames.

---

## 💡 How It Works

- **YOLOv8** detects people (`class 0`) in every frame.
- **Deep SORT** assigns each person a unique, persistent ID.
- The output video overlays bounding boxes and labels on each tracked player.

---

## 🧯 Troubleshooting

- **ModuleNotFoundError** → Rerun the `pip install` commands.
- **Black window or crash** → Ensure the video file name is correct and valid.
- **Custom model load error** → Model may not match YOLO version. Use `yolov8n.pt` or consult your mentor.
- **Empty output video** → Increase confidence threshold or check if class filtering is too strict.

---

## 📬 Credits

Developed as part of an AI Internship Assignment.  
Thanks to Ultralytics (YOLOv8) and the Deep SORT Realtime open-source contributors.


