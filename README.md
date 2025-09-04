# SignBot_Deploy
SignBot_Deploy is a real-time gesture recognition system that processes webcam input and makes directional predictions using deep learning models (including COSMOS). This README will guide you through setting up the environment, downloading pre-trained models, and running the demo.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12.10**
- **Git**
- (Optional) NVIDIA GPU + CUDA drivers for GPU acceleration

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kahleeeb3/SignBot_Deploy.git
cd SignBot_Deploy
```

### 2. Create a Virtual Environment
```bash
py --list
py -3.8 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate
python -m pip install --upgrade pip
```

### 3. Download & Extract Model Files
```bash
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1AeiNM1UlNlTfl5gWVhonYw2yPyUcfnvs/view?usp=sharing -O modules/signbot_demo.zip
python -m zipfile -e modules/signbot_demo.zip modules
del modules/signbot_demo.zip
```

<!-- ### 4. Setup COSMOS Tokenizer
```bash
cd modules/Cosmos-Tokenizer
pip install -r requirements.txt
pip install -e .
cd ../..
``` -->

### 5. Things that need fixed
```python
# modules\pose_hand_landmark_code\MediapipeLandmarks.py line 8
import modules.pose_hand_landmark_code.drawing_styles as dr_styles
```

### 6. Install Python Packages
```bash
pip cache purge
pip install -r requirements.txt

# Check that CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

## ▶️ Usage
After installation, run the main demo script:
```bash
python main_robot_test.py
```

## Docker
```bash
wsl -d Ubuntu
sudo docker build -t signbot_image --platform linux/arm64 -f Extension/Dockerfile.l4t .
```
```bash
sudo docker image inspect signbot_image --format '{{.Os}}/{{.Architecture}}'
```
```bash
sudo docker save signbot_image | pigz > Extension/signbot.tgz
```