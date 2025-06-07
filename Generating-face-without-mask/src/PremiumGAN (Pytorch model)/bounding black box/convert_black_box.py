import cv2
import numpy as np
from ultralytics import YOLO

def convert(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(source=img, save=False)

    if results:
        boxes = results[0].boxes  

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy() 
            x1, y1, x2, y2 = map(int, xyxy) 

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=-1)

    return img_rgb 