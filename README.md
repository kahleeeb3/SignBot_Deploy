# SignBot_Deploy

SignBot_Deploy is a real-time gesture recognition system that processes webcam input and makes directional predictions using deep learning models (including COSMOS). This README will guide you through setting up the environment, downloading pretrained models, and running the demo.

---

## 🚀 Getting Started

### Prerequisites

- **Conda** (Anaconda or Miniconda)
- **Git**
- (Optional) NVIDIA GPU + CUDA drivers for GPU acceleration

---

## 📦 Installation

### 1. Clone the Repository

```
git clone https://github.com/tonoypodder/SignBot_Deploy.git
cd SignBot_Deploy
```

### 2. 🐍 Create the Conda Environment
Make sure you have Conda installed.
```
conda env create -f environment.yml
conda activate SignBot_Deploy
```

### 3. Upgrade Pip & Install Python Packages
```
python -m pip install --upgrade pip
pip install gdown opencv-python mediapipe natsort
```

### 4. Install GPU-Compatible PyTorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install PyTorch Lightning
```
conda install lightning -c conda-forge
```

### 6. Download & Extract Model Files
Download the pretrained models ZIP:
```
gdown --fuzzy https://drive.google.com/file/d/1AeiNM1UlNlTfl5gWVhonYw2yPyUcfnvs/view?usp=sharing
```
Extract the archive:
```
python -m zipfile -e signbot_demo.zip ./
```

### 7. Setup COSMOS Tokenizer
```
cd Cosmos-Tokenizer
pip install -r requirements.txt
pip install -e .
cd ..
```

## ▶️ Usage
After installation, run the main demo script:
```
python main_robot_test.py
```

## 📝 Notes
If you encounter operator torchvision::nms does not exist, ensure torch and torchvision are installed from the same source and are version-compatible.

The GPU-enabled PyTorch build will fall back to CPU if no GPU is available.

Python 3.8+ is required (3.10 recommended).
