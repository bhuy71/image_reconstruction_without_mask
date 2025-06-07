import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import re
import cv2
from PIL import Image
import torchvision.transforms as transforms
class MaskDataset(Dataset):
        def __init__(self, with_mask_paths, transform=None):
            self.with_mask_paths = with_mask_paths
            self.transform = transform

        def __len__(self):
            return len(self.with_mask_paths)

        def __getitem__(self, idx):
            # Đọc ảnh không có khẩu trang và có khẩu trang
            with_mask_img = Image.open(self.with_mask_paths[idx]).convert('RGB')

            # Áp dụng biến đổi nếu có
            if self.transform:
                with_mask_img = self.transform(with_mask_img)

            return with_mask_img

def load_image(img_path):
    # Các tham số
    SIZE = 128  # Kích thước ảnh đầu vào

    # Các phép biến đổi (resize và chuẩn hóa)
    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),   # Resize ảnh về kích thước 64x64
        transforms.ToTensor(),             # Chuyển ảnh thành tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa về [-1, 1]
    ])

    # Tạo dataset và dataloader
    img = MaskDataset([img_path], transform=transform)

    # Tạo DataLoader với shuffle và chia batch
    img_loader=DataLoader(img)
    return img_loader

