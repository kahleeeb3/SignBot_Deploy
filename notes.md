# SignBot_Deploy
SignBot_Deploy is a real-time gesture recognition system that processes webcam input and makes directional predictions using deep learning models (including COSMOS). This README will guide you through setting up the environment, downloading pretrained models, and running the demo.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12.10**
- **Git**
- (Optional) NVIDIA GPU + CUDA drivers for GPU acceleration

---

## 📦 Installation

### 1. Clone the Repository
```
git clone https://github.com/tonoypodder/SignBot_Deploy.git
cd SignBot_Deploy
```

### 2. Create a Virtual Environment
```
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate
python -m pip install --upgrade pip
```

## gdown
```
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1AeiNM1UlNlTfl5gWVhonYw2yPyUcfnvs/view?usp=sharing -O modules/signbot_demo.zip
python -m zipfile -e modules/signbot_demo.zip modules
```
## Cosmos-Tokenizer
```
cd modules/Cosmos-Tokenizer
pip install -r requirements.txt
pip install -e .
cd ../..
```
## pips
```
pip install opencv-python
pip install lightning
pip install torchvision
pip install mediapipe
pip install natsort
```


## ▶️ Usage
After installation, run the main demo script:
```
python main_robot_test.py
```
<!-- Install dependencies in the venv
```
pip install -r requirements.txt
``` -->


# THINGS THAT NEED FIXED
```python
# modules\pose_hand_landmark_code\MediapipeLandmarks.py line 8 and 
import modules.pose_hand_landmark_code.drawing_styles as dr_styles
```