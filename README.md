# 🧠 YOLO Object Detection 🔍

A Python-based implementation of **YOLO (You Only Look Once)** for real-time object detection in images and video using OpenCV and pre-trained YOLO models.

---

## 📌 Features

- 🎯 Real-time object detection using YOLOv3 or YOLOv5
- 🖼️ Detects multiple object classes (COCO dataset)
- 📹 Works with images, webcam, or video files
- 📦 Pre-trained weights support (Darknet / PyTorch)
- 📊 Displays bounding boxes and class labels with confidence scores

---

## 🛠️ Technologies Used

- **Python 3.x**
- `OpenCV` – for image/video processing
- `NumPy` – for matrix operations
- `YOLOv3` / `YOLOv5` – pre-trained model weights
- `PyTorch` (if using YOLOv5)
- `Darknet` format (if using YOLOv3)

---

## 📂 Project Structure

```bash
YOLO/
├── yolov3.cfg             # YOLOv3 model config
├── yolov3.weights         # YOLOv3 pre-trained weights
├── coco.names             # Class labels (80 COCO classes)
├── detect.py              # Python script for detection
├── input.jpg              # Sample input image
├── output.jpg             # Output with bounding boxes
├── video_input.mp4        # Sample video input
├── video_output.mp4       # Processed video
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
🚀 Getting Started
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/JJIShanid/YOLO.git
cd YOLO
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install opencv-python numpy
🔍 Object Detection on Image
bash
Copy
Edit
python detect.py --image input.jpg --output output.jpg
🔍 Detection on Webcam
bash
Copy
Edit
python detect.py --webcam
📹 Detection on Video
bash
Copy
Edit
python detect.py --video video_input.mp4 --output video_output.mp4
⚙️ Parameters (via argparse)
Argument	Description
--image	Path to input image
--video	Path to input video
--output	Path to save the output
--webcam	Use webcam as input
--confidence	Minimum confidence threshold (default: 0.5)
--threshold	Non-maxima suppression threshold (default: 0.4)

🧠 How YOLO Works
YOLO (You Only Look Once) divides the image into a grid and simultaneously predicts:

Bounding boxes

Class labels

Confidence scores

This results in fast and accurate real-time detection, suitable for embedded systems, robotics, and surveillance.

📈 Sample Output
Input Image	Detection Result

Note: Add screenshots if available.

🧪 YOLO Versions
YOLOv3: Based on Darknet (uses .cfg and .weights)

YOLOv5: Based on PyTorch (uses .pt models — optional)

You can download the official YOLOv3 weights from:
https://pjreddie.com/media/files/yolov3.weights

📄 License
This project is licensed under the MIT License.

✍️ Author
Ishan JJIShanid

🙌 Acknowledgements
YOLO: Real-Time Object Detection

COCO Dataset

Ultralytics YOLOv5
