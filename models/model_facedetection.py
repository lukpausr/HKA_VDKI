# model_transferlearning.py

"""
This module contains the implementation of models for face detection and transfer learning.
Models and their suitability:
- EfficientNet-B3 to B5: High resolution and excellent feature extraction, ideal for fine distinctions.
- ResNet-101: Deeper model, capable of recognizing small details.
- DenseNet121/201: Particularly effective for small differences between classes.
- optional: FaceNet / ArcFace (customized): Ideal for "Face Embedding" when measuring similarities later.
"""

# required imports
import torch
from torch import nn
import pytorch_lightning as pl

# Pytorch Image Models
import timm