import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm as tqdm
import os,re,cv2
from tensorflow.keras.utils import img_to_array
from ultralytics import YOLO
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

# defining the size of the image
SIZE = 256
without_mask = []
with_mask=[]
NoofIMGs = 9001

path_1 = '/kaggle/input/my-data1/data/with_mask/train'
path_2 ='/kaggle/input/my-data1/data/without_mask/train'
files_1 = sorted_alphanumeric(os.listdir(path_1))
files_2 = sorted_alphanumeric(os.listdir(path_2))

for i in tqdm(files_1,total=NoofIMGs):
    if i == str(NoofIMGs)+'.png':
        break
    else:
        img = cv2.imread(path_1 + '/'+i,1)

        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))

        #normalising image
        img = (img - 127.5) / 127.5

        img = img.astype(float)

        without_mask.append(img_to_array(img))
for i in tqdm(files_2,total=NoofIMGs):
    if i == str(NoofIMGs)+'.png':
        break
    else:
        img = cv2.imread(path_2 + '/'+i,1)

        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))

        #normalising image
        img = (img - 127.5) / 127.5

        img = img.astype(float)

        with_mask.append(img_to_array(img))
# Discriminator Module


class ResidualBlock(Model):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x + inputs)

class SpatialAttentionModule(Model):
    def __init__(self, filters):
        super(SpatialAttentionModule, self).__init__()
        self.conv_wu = layers.Conv2D(1, kernel_size=3, padding='same')
        self.conv_wi = layers.Conv2D(1, kernel_size=3, padding='same')
        self.conv_wd = layers.Conv2D(1, kernel_size=3, padding='same')
        self.conv_wx = layers.Conv2D(1, kernel_size=3, padding='same')

    def call(self, f_u, f_i, f_d, f_x):
        wu = self.conv_wu(f_u)
        wi = self.conv_wi(f_i)
        wd = self.conv_wd(f_d)
        wx = self.conv_wx(f_x)
        attention_map = wu * wi * wd * wx
        return tf.reduce_sum(tf.abs(attention_map), axis=-1, keepdims=True)

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.residual_block1 = ResidualBlock(64)
        self.residual_block2 = ResidualBlock(128)
        self.spatial_attention1 = SpatialAttentionModule(64)
        self.spatial_attention2 = SpatialAttentionModule(128)

        self.conv1 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')
        self.conv2 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')
        self.conv3 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')
        self.conv4 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')
        self.conv5 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')
        self.conv6 = layers.Conv2D(1, kernel_size=4, strides=2, padding='same')

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Residual Block + Attention Block 1
        f_u1 = self.residual_block1(inputs)
        attention_map1 = self.spatial_attention1(f_u1, f_u1, f_u1, f_u1)
        x = tf.nn.relu(self.conv1(attention_map1))

        # Residual Block + Attention Block 2
        f_u2 = self.residual_block2(x)
        attention_map2 = self.spatial_attention2(f_u2, f_u2, f_u2, f_u2)
        x = tf.nn.relu(self.conv2(attention_map2))

        x = tf.nn.relu(self.conv3(x))
        x = tf.nn.relu(self.conv4(x))
        x = tf.nn.relu(self.conv5(x))
        x = tf.nn.sigmoid(self.conv6(x))

        return x

latent_dim = 100

def Generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(256 * 256 * 3, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.Reshape((256, 256, 3)))

    # Encoder
    model.add(layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.LeakyReLU())

    # Decoder
    model.add(layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, 4, strides=1, padding='same', activation='tanh'))

    return model
# Optimizer

initial_learning_rate = 0.0001
decay_rate = 1e-8
decay_steps = 10000

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, clipvalue=1.0)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
buffer_size = NoofIMGs + int(NoofIMGs / 10)
batch_size = 32

combined_dataset = tf.data.Dataset.from_tensor_slices((np.array(without_mask), np.array(with_mask)))
combined_dataset = combined_dataset.shuffle(buffer_size).batch(batch_size)
def generator_loss(discriminator_y_output, discriminator_x_output, fake_x_output, real_y_output):
    loss_gan_y = cross_entropy(tf.ones_like(discriminator_y_output), discriminator_y_output)
    loss_gan_x = cross_entropy(tf.ones_like(discriminator_x_output), discriminator_x_output)

    return loss_gan_y + loss_gan_x

def discriminator_loss(discriminator_x_output, discriminator_y_output, real_x_output, real_y_output):
    real_loss_y = cross_entropy(tf.ones_like(real_y_output), real_y_output)
    fake_loss_y = cross_entropy(tf.zeros_like(discriminator_y_output), discriminator_y_output)
    loss_dis_y = real_loss_y + fake_loss_y

    real_loss_x = cross_entropy(tf.ones_like(real_x_output), real_x_output)
    fake_loss_x = cross_entropy(tf.zeros_like(discriminator_x_output), discriminator_x_output)
    loss_dis_x = real_loss_x + fake_loss_x

    return loss_dis_y + loss_dis_x
def convert(image_array, model):
    img_rgb = (image_array * 127.5 + 127.5).astype(np.uint8)

    results = model.predict(source=img_rgb, save=False)

    if results:
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy() 
            x1, y1, x2, y2 = map(int, xyxy) 

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=-1)
    
    img_rgb = (img_rgb - 127.5) / 127.5
    img_rgb = img_rgb.astype(float)
    
    return img_to_array(img_rgb)
def apply_bounding_box_mask(image_batch):
    model = YOLO('/kaggle/input/mask_detection/other/default/1/mask_detection.pt')
    processed_images = []
    for image in image_batch:
        masked_image = convert(image, model)
        processed_images.append(masked_image)
    return np.array(processed_images)
def train_step(without_mask, with_mask, generator, discriminator, optimizer):
    images_with_mask = apply_bounding_box_mask(with_mask)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(images_with_mask, training=True)
        
        fake_output_y = discriminator(generated_images, training=True)
        real_output_y = discriminator(without_mask, training=True)

        fake_output_x = discriminator(generator(real_output_y), training=True)
        real_output_x = discriminator(images_with_mask, training=True)

        gen_loss = generator_loss(fake_output_y, fake_output_x, without_mask, with_mask)
        dis_loss = discriminator_loss(fake_output_y, fake_output_x, real_output_y, real_output_x)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)

    optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return {'gen_loss': gen_loss, 'disc_loss': dis_loss}
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, generator=Generator(), discriminator=Discriminator())
best_gen_loss = float('inf')
best_disc_loss = float('inf')
def train(epochs, combined_dataset, generator, discriminator, optimizer):
    global best_gen_loss, best_disc_loss
    for epoch in range(epochs):
        start = time.time()
        print("Epoch :", epoch + 1)
        combined_dataset = combined_dataset.shuffle(buffer_size)

        for without_mask_batch, with_mask_batch in tqdm(combined_dataset):
            loss = train_step(without_mask_batch, with_mask_batch)

        # Kiểm tra và lưu checkpoint nếu loss giảm
        if loss['gen_loss'] < best_gen_loss and loss['disc_loss'] < best_disc_loss:
            best_gen_loss = loss['gen_loss']
            best_disc_loss = loss['disc_loss']
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Checkpoint saved at epoch {epoch + 1} with Generator Loss: {best_gen_loss} | Discriminator Loss: {best_disc_loss}")

        print(f"Time: {np.round(time.time() - start, 3)} secs")
        print(f"Generator Loss: {loss['gen_loss']} | Discriminator Loss: {loss['disc_loss']}")
        print("\n")
    
generator = Generator()
discriminator = Discriminator()

train(epochs=50, combined_dataset=combined_dataset, generator=generator, discriminator=discriminator, optimizer=optimizer)