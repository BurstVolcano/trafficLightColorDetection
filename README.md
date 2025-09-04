# Traffic Light Detection System

A traffic light state detection system using YOLOv8 and HSV color space analysis.
The system can detect red, yellow, and green lights from videos or live webcam feeds.
It supports both command-line use and a Streamlit web application.

---

## Features

* Real-time detection from webcam or video files
* YOLOv8 object detection for traffic light localization
* HSV color space analysis to classify states
* Video display with bounding boxes and labels
* Command line interface with options for input and output
* Streamlit web app for easy video upload and analysis

---

## Requirements

### Python dependencies

Install using pip:

```
pip install -r requirements.txt
```

### System dependencies

On Linux or Streamlit Cloud, add a file named `packages.txt` with:

```
libgl1
libglib2.0-0
```

### Tested on

* Python 3.8+
* OpenCV 4.x
* PyTorch (installed with ultralytics)
* YOLOv8n model (auto downloaded on first run)

---

## Installation

1. Clone the repository:

```
git clone https://github.com/BurstVolcano/trafficLightColorDetection/tree/master
cd traffic-light-detection
```

2. Install Python dependencies:

```
pip install -r requirements.txt
```

3. (Optional) Install system libraries if you get libGL errors:

```
sudo apt-get install -y libgl1 libglib2.0-0
```

---

## Usage

### Command line

Run with default webcam:

```
python main.py
```

Run with a video file:

```
python main.py -v path/to/video.mp4
```

Use a specific camera index:

```
python main.py -c 1
```

Save processed video:

```
python main.py -v input.mp4 -s output.mp4
```

---

### Streamlit web app

Start the web app:

```
streamlit run streamlitapp.py
```

The app allows you to upload video files, adjust settings such as confidence threshold and frame skip, and view processed results in your browser.

---

## How it works

1. YOLOv8 detects traffic lights in each frame.
2. Detected lights are cropped to a standard size.
3. Each crop is split into three zones: top (red), middle (yellow), bottom (green).
4. HSV masks are applied to count pixels of each color.
5. The zone with enough colored pixels determines the light state.
6. Results are drawn on the video frames with bounding boxes and labels.

---

## Project structure

```
traffic-light-detection/
│
├── traffic_detector.py     
├── video_processor.py     
├── main.py                
├── streamlitapp.py       
├── requirements.txt
├── packages.txt            
└── README.md
```

---

## Example output

* Bounding boxes drawn around detected traffic lights
* Labels showing state and confidence
* Streamlit app shows video frames and detection statistics

---

## Performance

* 20 to 40 FPS on CPU
* Faster with GPU support in PyTorch
* Works best with clear daylight videos and standard traffic lights

---

## Limitations

* May fail in poor lighting or unusual traffic light designs
* Accuracy depends on video quality
* Intended for research and education, not production use

---

## Acknowledgments

* Ultralytics YOLOv8
* OpenCV
* PyTorch

---

## License

This project is licensed under the MIT License.