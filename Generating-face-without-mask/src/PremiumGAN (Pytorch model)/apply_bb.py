import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
import contextlib
import io
from PIL import Image
import torchvision.transforms as transforms

# Hàm chuyển đổi ảnh tensor sang dạng có bounding box
def convert(image_tensor, model):
    # Chuyển từ [-1, 1] về [0, 255] và chuyển từ tensor PyTorch sang NumPy
    img_original = ((image_tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    
    if img_original.shape[2] != 3:  # Kiểm tra số kênh của ảnh
        raise ValueError(f"Invalid image format: Expected 3 channels, but got {img_original.shape}")

    # Chuyển từ RGB sang BGR và resize cho YOLO
    img_rgb_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_rgb_original, (640, 640))

    # Thực hiện dự đoán với mô hình YOLO
    with contextlib.redirect_stdout(io.StringIO()):
        results = model(img_resized, verbose=False)

    # Xử lý bounding box từ YOLO
    if isinstance(results, list) and len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1 = int(x1 * (128 / 640))
            y1 = int(y1 * (128 / 640))
            x2 = int(x2 * (128 / 640))
            y2 = int(y2 * (128 / 640))
            img_rgb_original = cv2.rectangle(img_rgb_original, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=-1)
    else:
        raise ValueError("The result does not contain valid bounding boxes.")

    # Chuyển lại ảnh về định dạng RGB
    img_rgb_display = cv2.cvtColor(img_rgb_original, cv2.COLOR_BGR2RGB)
    return img_rgb_display

# Hàm áp dụng mask bounding box cho một batch ảnh
def apply_bounding_box_mask(image_batch, model):
    processed_images = []
    for image_tensor in image_batch:
        masked_image = convert(image_tensor, model)
        processed_images.append(masked_image)

    # Chuyển danh sách ảnh về tensor PyTorch
    processed_images = np.array(processed_images)
    processed_images = torch.from_numpy(processed_images).float()

    # Chuẩn hóa lại các ảnh về phạm vi [-1, 1]
    processed_images = (processed_images / 127.5) - 1
    return processed_images