# Unmask System. Reconsstructing Faces Beyond the Mask


![Logo](https://github.com/user-attachments/assets/9f9680e0-e058-4fa0-983c-972cb8ba9451)

***


## Introduction

In an era where public health and safety are paramount, understanding and detecting the presence of face masks has become a crucial task.

This project represents my initiative to develop a model capable of detecting face masks in images. Leveraging the **Face Mask Lite** dataset, I designed a machine learning pipeline focused on efficiency and accuracy. My system utilizes **Python** as the primary programming language, integrating popular libraries like Keras and Pytorch to build and train my detection model.

The application is lightweight, scalable, and optimized for practical use cases such as public monitoring systems and workplace compliance checks. It prioritizes accuracy while ensuring computational efficiency, making it adaptable for deployment in real-world scenarios.

***This project is intended for `educational purposes only`. The model's performance is subject to dataset limitations and does not guarantee flawless detection. It should not be used as a replacement for professional safety measures or health compliance systems.***

***


## Key features

- Data preprocessing and **exploratory data analysis** performed in Jupyter Notebooks with support from [Pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/).
- **GAN-based model** implemented for mask removal and face image generation using [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).
- **Face mask detection** integrated with [UNet](https://arxiv.org/abs/1505.04597) architecture for accurate segmentation and mask identification.
- **Diffusion model** integrated for inpainting and refining generated images, utilizing pre-trained weights from [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/).
- Lightweight **GUI** built with Python using [Tkinter](https://docs.python.org/3/library/tkinter.html) for an interactive user experience.
- Deployment-ready script for model inference, optimized with [Flask](https://flask.palletsprojects.com/en/3.0.x/) to serve predictions via HTTP API.

***


## Project Structure

- **BasicGAN (Keras model)**:  
   Directory containing the implementation of the BasicGAN using Keras.

- **PremiumGAN (Pytorch model)**:  
   Directory containing the implementation of PremiumGAN  built with PyTorch.

- **Pretrained.StableDiffusion2Inpainting**:  
   Directory storing pretrained weights and configurations for Stable Diffusion Inpainting models.

- **Exploratory Data Analysis.ipynb**:  
   Jupyter Notebook performing **Exploratory Data Analysis** to understand, clean, and preprocess the dataset.

- **infer.py**:  
   Python script for performing inference using the trained models to reconstruct images.

- **inferGUI.ipynb/ inferGUI.py**:  
   Jupyter Noteboothonk and Py implementing a GUI-based inference system for testing the model outputs interactively.

- **install.sh**:  
   Shell script for setting up the environment and installing dependencies for the project.

- **output.png** / **with-mask-default-mask-seed0008.png**:  
   Example output images showing input masked faces and corresponding reconstructed results.

- **README.md**:  
   Main documentation file explaining the project setup, structure, and usage instructions.
  
***


## Requirements
Referring to my libraries use: `install.sh`

***


## Installation and usage

Please refer to these following links for essential document:

[Dataset Link](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset)

[Training dataset](https://drive.google.com/drive/folders/1YSau5CWdgtpQGOpCvqhKLqvBwnVrO7jw?usp=sharing)

[BasicGan checkpoint (Keras model)](https://drive.google.com/drive/folders/1EptsRKAHr3xJ31wTGbvBKaIuw1j2nTGu)

[Detection model](https://drive.google.com/drive/folders/1EptsRKAHr3xJ31wTGbvBKaIuw1j2nTGu)

[Premium GAN checpoint (Pytorch model)](https://drive.google.com/drive/folders/1JktC1krdN7wD1XqfuDTwnlsClhQaZ1Kg?usp=drive_link)

[Diffusion checkpoint](https://www.kaggle.com/models/bhuy71/diffusion)

### Note
Before you run infer.py or you use the GUI you must download all the model in the links above including (Keras model, Detection model, pytorch model, diffusion model). Then you must change the directory to the model in the file you want to run.

### infer.py

   1. To run with the PyTorch generator model:

     python script.py --model PyTorch Generator --pytorch_model_path /path/to/pytorch/model --input /path/to/input/image --output output.png

   2. To run with the Diffusion generator model:

     python script.py --model Diffusion Generator --diffusion_model_path /path/to/diffusion/model --input /path/to/input/image --output output.png

   3. To run with the Keras generator model:

     python script.py --model Keras Generator --keras_model_path /path/to/keras/model --input /path/to/input/image --output output.png

### GUI( recommended)
  
  Apart from creating the infer.py script following the teaching assistant's instructions, I have also implemented an additional user-friendly interface. This interface allows users to run my code more conveniently through an intuitive GUI. Following these steps to experiment my product:
  ### Note: I recommend you use Kaggle for faster processing because of its strong GPU. 

  
  I. If you use Kaggle:
  
        1.You create a new notebook then import notebook UnMaskUI.ipynb into it.
        
        2.You must upload the models you dowloaded in the links into your notebook and change the directory to the models in the notebook.
        
           + KERAS_MODEL_PATH = <path_to_keras_model>
           
           + DETECTION_MODEL_PATH = <path_to_detection_model>
           
           + PYTORCH_MODEL_PATH = <path_to_pytorch_model>
           
           + DIFFUSION_MODEL_PATH = <path_to_diffusion_model>
           
        3.Run all cells and in the final cell , there is a link of the interface appears, click on that to use my interface.
        
  II. If you do not use Kaggle:
  
     1. Direct and open the file: UnMaskUI.py or UnMaskUI.ipynb. They are all the same but displayed into 2 different format
     
     2. Paste the model path to the param:
        + KERAS_MODEL_PATH = <path_to_keras_model>
        + DETECTION_MODEL_PATH = <path_to_detection_model>
        + PYTORCH_MODEL_PATH = <path_to_pytorch_model>
        + DIFFUSION_MODEL_PATH = <path_to_diffusion_model>
     
     2. Run all the cell 
     
     3. After execute the last cell , there will be a link. You can experience by yourself now
  
***

## Acknowledgments
This project would not have been possible without the invaluable contributions of several open-source libraries, such as [Keras](link Keras), [PyTorch](link Pytorch), and [Diffusers](link Diffusers). Their robust tools and resources were instrumental in the success of this project.

I extend my heartfelt gratitude to my lecturers, Professor Tran Vinh Duc for assigning me this challenging yet captivating project. It has been an incredible learning opportunity that has significantly enhanced my knowledge and skillset.


***

## License
This project is licensed under the [MIT License](LICENSE).
