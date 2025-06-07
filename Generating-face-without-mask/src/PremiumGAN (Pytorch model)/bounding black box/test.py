import convert_black_box
from ultralytics import YOLO
import matplotlib.pyplot as plt
model=YOLO('C:/Sam C-Gan project/bounding black box/mask_detection.pt')
image_path='C:/Sam C-Gan project/bounding_box_data/train/images/images (2).jpg'
output_image=convert_black_box.convert(image_path,model)
plt.imshow(output_image)
plt.axis('off')
plt.show()
