import xml.etree.ElementTree as ET
import argparse
from skimage.transform import resize
from PIL import Image
import numpy as np
from numpy import shape
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from skimage import io, color
import mlflow
import random
import glob
from skimage.transform import resize
import skimage.transform as transform
import mlflow

config = tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.1 
session = tf.compat.v1.Session(config=config) 

def parse_args():

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--raw_data", type=str, help="Name of raw-data dataset")
    parser.add_argument("--prepared_X", type=str, help="Path to store prepared images")
    parser.add_argument("--prepared_y", type=str, help="Path to store prepared masks")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    # parser.add_argument("--enable_monitoring", type=str, help="enable logging to ADX")
    # parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")
    args = parser.parse_args()

    return args

def log_training_data(df, table_name):
    from obs.collector import Online_Collector
    collector = Online_Collector(table_name)
    collector.batch_collect(df)


args = parse_args()
# image_path = "road-segmentation/images"
image_path  = Path(args.raw_data) / 'images/*'
# mask_path  = "road-segmentation/masks"
mask_path   = Path(args.raw_data) / 'masks/*'

print(image_path)

# fig = plt.figure()

def display_image(image_folder):
    
    image_files = [f for f in (os.listdir(image_folder)) if f.endswith(('jpg','png','jpeg'))]
    
    #print(image_files)
    
    fig,axes = plt.subplots(3,3,figsize=(10,10))
    
    for i, file_name in enumerate(image_files):  
        
        if i>=9:
            break
        row = i//3
        col = i%3
        
         # Load and display the image in the current subplot
        image_path = os.path.join(image_folder, file_name)
        image = io.imread(image_path)
        axes[row, col].imshow(image)
        axes[row, col].set_title(file_name)
        axes[row, col].axis('off')
        
        
    # Ensure proper layout and show the plots
    plt.tight_layout()
    plt.savefig("images.png")
    mlflow.log_artifact("images.png")
    
    

image_folder = Path(args.raw_data) / 'images'
display_image(image_folder)

print(image_folder)

    # ----------------------------------------------------------

def mask_image(image_folder):
    
    image_files = [f for f in (os.listdir(image_folder)) if f.endswith(('jpg','png','jpeg'))]
    
    #print(image_files)
    
    fig,axes = plt.subplots(3,3,figsize=(10,10))
    
    for i, file_name in enumerate(image_files):  
        
        if i>=9:
            break
        row = i//3
        col = i%3
        
         # Load and display the image in the current subplot
        image_path = os.path.join(image_folder, file_name)
        image = io.imread(image_path)
        axes[row, col].imshow(image)
        axes[row, col].set_title(file_name)
        axes[row, col].axis('off')
        
        
    # Ensure proper layout and show the plots
    plt.tight_layout()
    plt.savefig("masks.png")
    mlflow.log_artifact("masks.png")
    

image_folder = Path(args.raw_data) / 'masks'
mask_image(image_folder)
 
image_data = []



# --------------------------------------------------------------------------------------------------------------

image_paths = sorted(glob.glob(str(image_path)), key=lambda x: x.split('.')[0])
mask_paths = sorted(glob.glob(str(mask_path)), key=lambda x: x.split('.')[0])

print(image_paths)

def resize_image(image, size):
    # Resize the image to the specified size
    resized_image = resize(image, size)
    return resized_image

def resize_mask(mask, size):
    # Convert the mask to grayscale
    mask = mask[:, :, :3] # remove the alpha channel from the image
    mask_gray = color.rgb2gray(mask)
    resized_mask = transform.resize(mask_gray, output_shape=size, order=0, mode='constant', anti_aliasing=False, cval=255)
    resized_mask = np.expand_dims(resized_mask, axis=2)
    return resized_mask

# Define the desired size
target_size = (512,512)
image_list = []
mask_list = []
for image_path, mask_path in zip(image_paths, mask_paths):
    # Load the image and mask
    image = plt.imread(image_path).astype(np.float32)
    mask = plt.imread(mask_path).astype(np.float32)

    # Resize the image and mask
    resized_image = resize_image(image, target_size)
    resized_mask = resize_mask(mask, target_size)

    image_list.append(resized_image)
    mask_list.append(resized_mask)

print(image_list)

# Convert the image and mask lists to arrays
image_array = np.array(image_list)
mask_array = np.array(mask_list)
print(len(image_array))
print(len(mask_array))

X_train, X_test, y_train, y_test = train_test_split(image_array,mask_array, test_size=0.05, random_state=23)


def create_unet_model(input_shape=(512, 512, 3)):
    # Define the input layer
    x = tf.keras.layers.Input(input_shape)

    # Encoder layers
    enc1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    enc2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(enc1)
    enc3 = tf.keras.layers.MaxPooling2D((2, 2))(enc2)
    enc3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(enc3)
    enc4 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(enc3)
    enc5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(enc4)
    enc6 = tf.keras.layers.MaxPooling2D((2, 2))(enc5)
    enc6 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(enc6)
    enc7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(enc6)
    enc8 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='relu')(enc7)

    # Skip connections
    skip1 = enc2
    skip2 = enc5
    # Decoder layers
    dec1 = tf.keras.layers.UpSampling2D((2, 2))(enc8)
    dec1 = tf.keras.layers.Concatenate()([dec1, skip2])
    dec1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(dec1)
    dec1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(dec1)

    dec2 = tf.keras.layers.UpSampling2D((2, 2))(dec1)
    dec2 = tf.keras.layers.Concatenate()([dec2, skip1])
    dec2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(dec2)
    dec2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(dec2)

    # Final decoder layer with sigmoid activation for binary output
    dec3 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(dec2)

    # Create and return the model
    model = tf.keras.Model(inputs=x, outputs=dec3)
    return model


model = create_unet_model()



loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=4, epochs=5,validation_data=(X_test, y_test)) 
# history = model.fit(X_train,y_train, batch_size=16, epochs=5,validation_data=(X_test, y_test)) 


# pickle.dump(model, open((Path("../local/data/prepared/") / "y_test.csv"), "wb"))
# pickle.dump(model, open((Path("../local/data/prepared/") / "X_test"), "wb"))

figure, axes = plt.subplots(3,3, figsize=(20,20))

for i in range(0,3):
    rand_num = random.randint(0,1)
    original_img = X_test[rand_num]
    # cv2.imwrite("new"+ str(i) + ".png",original_img)
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Original Image')
    
    original_mask = y_test[rand_num]
    axes[i,1].imshow(original_mask)
    axes[i,1].title.set_text('Original Mask')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted_mask = model.predict(original_img).reshape(512,512)
    axes[i,2].imshow(predicted_mask)
    axes[i,2].title.set_text('Predicted Mask')

plt.tight_layout()
plt.savefig("segmentation.png")
mlflow.log_artifact("segmentation.png")



x_path = (Path(args.prepared_X) / "X_test.npy")
np.save(x_path,X_test)
y_path = (Path(args.prepared_y) / "y_test.npy")
np.save(y_path,y_test)

model.save(Path(args.model_output) / "model.h5")






# if (args.enable_monitoring.lower == 'true' or args.enable_monitoring == '1' or args.enable_monitoring.lower == 'yes'):
#         log_training_data(args.raw_data, args.table_name)


# ----------------------------------------------------
