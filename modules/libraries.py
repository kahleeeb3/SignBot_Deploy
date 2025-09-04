import os
import glob
import pytorch_lightning as pl
# from torch import optim
# import pytorchvideo.data
# import torch.utils.data
import torch.nn as nn
import torchvision
# import natsort
# import random
# from transformers import ViTFeatureExtractor
# from sklearn.preprocessing import LabelEncoder
# from transformers import ViTMAEForPreTraining, ViTForImageClassification
import torchmetrics
from torchmetrics import Metric
# from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_lightning.callbacks import ModelCheckpoint
# import torchvision.models as torch_models
# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
# import timm
import yaml
import einops
from PIL import Image
import cv2
import math
import torch
# from cosmos_tokenizer.utils import (
#     numpy2tensor,
#     tensor2numpy,
# )
# from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import numpy as np
from modules.pose_hand_landmark_code.MediapipeLandmarks import HandDetectionModel, PoseDetectionModel