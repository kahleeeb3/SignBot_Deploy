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

### 3. Download & Extract Model Files
```
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1AeiNM1UlNlTfl5gWVhonYw2yPyUcfnvs/view?usp=sharing -O modules/signbot_demo.zip
python -m zipfile -e modules/signbot_demo.zip modules
```
### 4. Setup COSMOS Tokenizer
```
cd modules/Cosmos-Tokenizer
pip install -r requirements.txt
pip install -e .
cd ../..
```

## 5. Things that need fixed
```python
# modules\pose_hand_landmark_code\MediapipeLandmarks.py line 8
import modules.pose_hand_landmark_code.drawing_styles as dr_styles
```

### 6. Install Python Packages
```
pip install -r requirements.txt
```

## ▶️ Usage
After installation, run the main demo script:
```
python main_robot_test.py
```