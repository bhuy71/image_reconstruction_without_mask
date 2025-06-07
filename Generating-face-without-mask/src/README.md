# Unmask System. Reconsstructing Faces Beyond the Mask — Group 16


![Logo](https://github.com/user-attachments/assets/9f9680e0-e058-4fa0-983c-972cb8ba9451)

***


## Introduction

In an era where public health and safety are paramount, understanding and detecting the presence of face masks has become a crucial task.

This project represents our team’s initiative to develop a model capable of detecting face masks in images. Leveraging the **Face Mask Lite** dataset, we designed a machine learning pipeline focused on efficiency and accuracy. Our system utilizes **Python** as the primary programming language, integrating popular libraries like Keras and Pytorch to build and train our detection model.

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
Referring to our libraries use: `install.sh`

***


## Installation and usage

Please refer to these following links for essential document:

[Dataset Link](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset)

[Training dataset](https://drive.google.com/drive/folders/1YSau5CWdgtpQGOpCvqhKLqvBwnVrO7jw?usp=sharing)

[BasicGan checkpoint (Keras model)](https://drive.google.com/file/d/1vjCB1Q21YFjnUDGkui1DrJsFLC7ztLlc/view?usp=sharing)

[Detection model](https://drive.google.com/file/d/1VsfW6QPsrOQqxsaV-ER3W2rBfryzE2P1/view?usp=sharing)

[Premium GAN checpoint (Pytorch model)](https://drive.google.com/drive/folders/1JktC1krdN7wD1XqfuDTwnlsClhQaZ1Kg?usp=drive_link)

[Diffusion checkpoint](https://drive.google.com/file/d/186KQQTm-MmXlFYh1MzB2NUWw5KZsRqWk/view?usp=sharing)

### infer.py

   1. To run with the PyTorch generator model:

     python script.py --model PyTorch Generator --pytorch_model_path /path/to/pytorch/model --input /path/to/input/image --output output.png

   2. To run with the Diffusion generator model:

     python script.py --model Diffusion Generator --diffusion_model_path /path/to/diffusion/model --input /path/to/input/image --output output.png

   3. To run with the Keras generator model:

     python script.py --model Keras Generator --keras_model_path /path/to/keras/model --input /path/to/input/image --output output.png

### GUI
  
  Apart from creating the infer.py script following the teaching assistant's instructions, we have also implemented an additional user-friendly interface. This interface allows users to run our code more conveniently through an intuitive GUI. Following these steps to experiment our product:
  1. Direct and open the file: UnMaskUI.py or UnMaskUI.ipynb. They are all the same but displayed into 2 different format
  
  2. Paste the model path to the param:
     + KERAS_MODEL_PATH = <path_to_keras_model>
     + DETECTION_MODEL_PATH = <path_to_detection_model>
     + PYTORCH_MODEL_PATH = <path_to_pytorch_model>
     + DIFFUSION_MODEL_PATH = <path_to_diffusion_model>
  
  2. Run all the cell 
  
  3. After execute the last cell , there will be a link. You can experience by yourself now
  
***


## Demo
![Demo](https://github.com/user-attachments/assets/51e5dfe0-7586-47ba-92c2-5a233c3fca9f)

***


## Acknowledgments
This project would not have been possible without the invaluable contributions of several open-source libraries, such as [Keras](link Keras), [PyTorch](link Pytorch), and [Diffusers](link Diffusers). Their robust tools and resources were instrumental in the success of this project.

We extend our heartfelt gratitude to our lecturers, Professor Nguyen Hung Son and Professor Trang Viet Chung, for assigning us this challenging yet captivating project. It has been an incredible learning opportunity that has significantly enhanced our knowledge and skillset.

Our sincere thanks also go to our teaching assistants, Doan The Vinh and Nguyen Ba Thiem, for their unwavering support and constructive feedback throughout the project. 

Finally, we would like to acknowledge our peers for their indirect contributions, offering both moral and practical support that kept us motivated and

***


## Contributors
- Lại Trí Dũng - 20225486
- Đỗ Đình Hoàng - 20225445
- Bùi Văn Huy - 20225497
- Vũ Việt Long - 20225508
- Trịnh Huynh Sơn - 20225526

***


## License
This project is licensed under the [MIT License](LICENSE).
