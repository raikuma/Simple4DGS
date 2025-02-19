import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack):
        self.viewpoint_stack = viewpoint_stack
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]

        resized_image_rgb = PILtoTorch(Image.open(viewpoint_cam.image_path), viewpoint_cam.resolution)
        viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            
        return viewpoint_image, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
