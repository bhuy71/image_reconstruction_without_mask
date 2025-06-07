#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os






# In[13]:


import torch
import argparse	
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import re
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import gradio as gr
from ultralytics import YOLO
import contextlib
import io
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf
#
# KERAS_MODEL_PATH="/kaggle/input/generator/tensorflow2/default/1/generator (1).h5"
# PYTORCH_MODEL_PATH='/kaggle/input/generator_epoch/pytorch/default/1/generator_epoch_60.pth'
# DETECTION_MODEL_PATH="/kaggle/input/mark_detection/pytorch/default/1/mask_detection.pt"
# DIFFUSION_MODEL_PATH="/kaggle/input/unmasking-diffusion/other/default/1/unmasking_diffusion.kitties015.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Double Convolutional Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.conv_1 = DoubleConv(3, 64)  # 64x128x128
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64x64

        self.conv_2 = DoubleConv(64, 128)  # 128x64x64
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x32x32

        self.conv_3 = DoubleConv(128, 256)  # 256x32x32
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x16x16

        self.conv_4 = DoubleConv(256, 512)  # 512x16x16
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512x8x8

        self.conv_5 = DoubleConv(512, 1024)  # 1024x8x8

        # Decoder
        self.upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 512x16x16
        self.conv_6 = DoubleConv(1024, 512)  # 512x16x16

        self.upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 256x32x32
        self.conv_7 = DoubleConv(512, 256)  # 256x32x32

        self.upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 128x64x64
        self.conv_8 = DoubleConv(256, 128)  # 128x64x64

        self.upconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 64x128x128
        self.conv_9 = DoubleConv(128, 64)  # 64x128x128

        self.output = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # 3x128x128

    def forward(self, batch):
        # Encoder
        conv_1_out = self.conv_1(batch)
        conv_2_out = self.conv_2(self.pool_1(conv_1_out))
        conv_3_out = self.conv_3(self.pool_2(conv_2_out))
        conv_4_out = self.conv_4(self.pool_3(conv_3_out))
        conv_5_out = self.conv_5(self.pool_4(conv_4_out))

        # Decoder
        conv_6_out = self.conv_6(torch.cat([self.upconv_1(conv_5_out), conv_4_out], dim=1))
        conv_7_out = self.conv_7(torch.cat([self.upconv_2(conv_6_out), conv_3_out], dim=1))
        conv_8_out = self.conv_8(torch.cat([self.upconv_3(conv_7_out), conv_2_out], dim=1))
        conv_9_out = self.conv_9(torch.cat([self.upconv_4(conv_8_out), conv_1_out], dim=1))

        # Output Layer
        output = self.output(conv_9_out)
        return torch.tanh(output)


# In[14]:


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
    transforms.Resize((SIZE, SIZE)),  # Resize ảnh về kích thước 64x64
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa về [-1, 1]
  ])

  # Tạo dataset và dataloader
  img = MaskDataset([img_path], transform=transform)

  # Tạo DataLoader với shuffle và chia batch
  img_loader = DataLoader(img)
  return img_loader



# In[15]:


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


# In[25]:


import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2
import contextlib
import io
from ultralytics import YOLO


def load_diffusion_model(model_path):
  pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")

  checkpoint = torch.load(model_path)
  pipe.unet.load_state_dict(checkpoint['unet'])
  pipe.vae.load_state_dict(checkpoint['vae'])
  pipe.text_encoder.load_state_dict(checkpoint['text_encoder'])

  if checkpoint.get('scheduler') is not None:
    pipe.scheduler.load_state_dict(checkpoint['scheduler'])

  pipe.to("cuda")
  return pipe


def convert_to_mask(image_tensor, model):
  img_original = ((image_tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)

  if img_original.shape[2] != 3:
    raise ValueError(f"Invalid image format: Expected 3 channels, but got {img_original.shape}")

  img_rgb_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
  img_resized = cv2.resize(img_rgb_original, (640, 640))

  with contextlib.redirect_stdout(io.StringIO()):
    results = model(img_resized, verbose=False)

  mask = np.zeros((128, 128), dtype=np.uint8)

  if isinstance(results, list) and len(results) > 0:
    boxes = results[0].boxes

    for box in boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

      x1 = int(x1 * (128 / 640))
      y1 = int(y1 * (128 / 640))
      x2 = int(x2 * (128 / 640))
      y2 = int(y2 * (128 / 640))

      mask[y1:y2, x1:x2] = 1
  else:
    raise ValueError("The result does not contain valid bounding boxes.")

  return mask


def generate_diffusion_image(pipe, input_image_path, mask, prompt):
  input_image = Image.open(input_image_path).convert("RGB")

  # Convert the binary mask to a PIL image
  mask_image = Image.fromarray((mask * 255).astype(np.uint8))

  # Generate output
  result = pipe(prompt=prompt, image=input_image, mask_image=mask_image)

  # Return the generated image
  return result.images[0]


def process_diffusion_image(model, detection_model, input_image_path, prompt="Restore the original image "):
  # Load the fine-tuned model

  input_image = Image.open(input_image_path).convert("RGB")
  preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
  ])
  image_tensor = preprocess(input_image) * 2 - 1

  mask = convert_to_mask(image_tensor, detection_model)

  output_image = generate_diffusion_image(model, input_image_path, mask, prompt)

  return output_image


# In[18]:




# In[19]:


from keras.preprocessing.image import img_to_array


def preprocess_image_to_latent(image_path):
  img = Image.open(image_path).convert('RGB')  # Đọc ảnh và chuyển sang RGB
  img = img.resize((128, 128))  # Resize ảnh
  img = np.array(img) / 127.5 - 1.0  # Chuẩn hóa về [-1, 1]
  img = tf.convert_to_tensor(img, dtype=tf.float32)
  img = tf.expand_dims(img, axis=0)
  return img


# In[20]:
detection_model = YOLO("PretrainedDiffusion_2_Inpainting/mask_detection.pt")
# Display and process images
def display_images(generator, model_yolo, img_loader, save_folder):
    """
    Process images and return a PIL format image.
    """
    with_mask_batch = next(iter(img_loader))
    with_mask_batch = with_mask_batch.to(device)  # Đảm bảo ảnh được chuyển sang GPU nếu có
    plt.figure(figsize=(15, 30))  # Tăng kích thước hiển thị (rộng x cao)

    os.makedirs(save_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    idx = 0  # Chỉ số của ảnh trong batch

    # 1. Hiển thị ảnh gốc từ dataloader (ảnh có khẩu trang)
    img_original = with_mask_batch[idx].cpu()  # Chuyển về CPU
    img_original = img_original * 0.5 + 0.5  # Bỏ chuẩn hóa, đưa ảnh về [0, 1]
    img_original = img_original.permute(1, 2, 0).numpy()  # Chuyển từ (C, H, W) -> (H, W, C)

    # 2. Áp dụng bounding box mask lên ảnh có khẩu trang
    images_with_mask = apply_bounding_box_mask(with_mask_batch, model_yolo)
    img_with_mask = images_with_mask[idx].cpu()  # Chuyển về CPU
    img_with_mask = img_with_mask * 0.5 + 0.5  # Bỏ chuẩn hóa
    img_with_mask = img_with_mask.numpy()  # Chuyển từ (C, H, W) -> (H, W, C)

    # 3. Truyền ảnh qua generator để sinh ảnh mới
    images_with_mask = images_with_mask.to(device)
    generated_images = generator(images_with_mask.permute(0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
    generated_image = generated_images[idx].cpu().detach()  # Chuyển về CPU
    generated_image = generated_image * 0.5 + 0.5  # Bỏ chuẩn hóa, đưa ảnh về [0, 1]

    # Hiển thị ảnh được sinh
    generated_img = generated_image.permute(1, 2, 0).cpu().detach().numpy()  # (C, H, W) -> (H, W, C)

# Log the shape and dtype for debugging
    print(f"Shape of generated_img: {generated_img.shape}, dtype: {generated_img.dtype}")

# Fix shape if necessary
    generated_img = np.squeeze(generated_img)  # Remove single dimensions if present

# Fix data type if necessary
    if generated_img.dtype != np.uint8:
      generated_img = (generated_img * 255).astype(np.uint8)  # Convert to uint8

# Convert to PIL image
    generated_image_pil = Image.fromarray(generated_img)

# Save or display the PIL image
    generated_image_pil.save("generated_image.png")


    return generated_image_pil  # Return the PIL image


# In[26]:



# Load models
def load_models(model_choice, pytorch_model_path, diffusion_model_path, keras_model_path, detection_model_path):
  if model_choice == "PyTorch Generator":
    pytorch_generator = Generator()
    pytorch_generator.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    pytorch_generator.to(device)
    pytorch_generator.eval()
    return pytorch_generator
  elif model_choice == "Diffusion Generator":
    return load_diffusion_model(diffusion_model_path)
  elif model_choice == "Keras Generator":
    return tf.keras.models.load_model(keras_model_path)
  else:
    raise ValueError("Invalid model name")


# Define argument parser
def parse_args():
  parser = argparse.ArgumentParser(description="Inference script for image generation.")
  parser.add_argument("--model", choices=["PyTorch Generator", "Diffusion Generator", "Keras Generator"], required=True,
                      help="Select the model for inference.")
  parser.add_argument("--pytorch_model_path", required=False, help="Path to the PyTorch model file.")
  parser.add_argument("--diffusion_model_path", required=False, help="Path to the Diffusion model file.")
  parser.add_argument("--keras_model_path", required=False, help="Path to the Keras model file.")
  parser.add_argument("--input", required=True, help="Path to the input image.")
  parser.add_argument("--output", default="output_image.png", help="Path to save the generated image.")
  return parser.parse_args()


# Main processing function
def main():
  args = parse_args()

  # Ensure you pass the correct model path based on the selected model
  model_choice = args.model
  if model_choice == "PyTorch Generator" and not args.pytorch_model_path:
    raise ValueError("PyTorch model path must be provided for PyTorch Generator.")
  if model_choice == "Diffusion Generator" and not args.diffusion_model_path:
    raise ValueError("Diffusion model path must be provided for Diffusion Generator.")
  if model_choice == "Keras Generator" and not args.keras_model_path:
    raise ValueError("Keras model path must be provided for Keras Generator.")

  # Load the selected model
  model = load_models(
    model_choice=model_choice,
    pytorch_model_path=args.pytorch_model_path,
    diffusion_model_path=args.diffusion_model_path,
    keras_model_path=args.keras_model_path,
    detection_model_path="PretrainedDiffusion_2_Inpainting/mask_detection.pt"  # This remains constant
  )

  # Assuming the YOLO model for detection
  yolo_model = YOLO("PretrainedDiffusion_2_Inpainting/mask_detection.pt")

  input_image_path = args.input
  output_image_path = args.output

  if model_choice == "PyTorch Generator":
    img_loader = load_image(input_image_path)
    output_image = display_images(model, yolo_model, img_loader, save_folder="result")
  elif model_choice == "Diffusion Generator":
    output_image = process_diffusion_image(model, yolo_model, input_image_path)
  elif model_choice == "Keras Generator":
    latent_vector = preprocess_image_to_latent(input_image_path)
    generated_image = model(latent_vector)
    generated_image = generated_image[0].numpy()
    generated_image = (generated_image + 1) / 2
    output_image = Image.fromarray((generated_image * 255).astype(np.uint8))
  else:
    raise ValueError("Invalid model name")

  output_image.save(output_image_path)
  print(f"Generated image saved to {output_image_path}")


if __name__ == "__main__":
  main()
