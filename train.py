import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import os
import random
import shutil
import numpy as np
import minio 
from io import BytesIO
from minio import Minio
from PIL import Image
from io import BytesIO
import random

def split_dataset_to_minio(main_bucket: str, subbuckets: list, train_ratio: float, minio_client: Minio):
    """
    Splits the dataset into train and test sets and uploads them to MinIO.

    Parameters:
    - main_bucket: The main bucket in MinIO containing the subbuckets.
    - subbuckets: List of subbuckets (e.g., ['real', 'fake', ...]).
    - train_ratio: Ratio for train/test split.
    - minio_client: Instance of Minio client.
    """
    
    for subbucket in subbuckets:
        objects = minio_client.list_objects( subbucket, recursive=True)
        files = [obj.object_name for obj in objects]
        random.shuffle(files)

        # Split the files into train and test sets
        train_files = files[:int(train_ratio * len(files))]
        test_files = files[int(train_ratio * len(files)):]

        # Copy train files to the train bucket
        train_folder = "dataset/train/" + subbucket
        for file_name in train_files:
            data = minio_client.get_object(main_bucket, file_name)
            img = Image.open(BytesIO(data.read()))
            img.verify()
            # Create the folder if it doesn't exist
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            
            #save the image
            img.save(train_folder + "/" + file_name.split("/")[-1])

            
        # Copy test files to the test bucket
        test_folder = "dataset/test/" + subbucket
        for file_name in test_files:
            data = minio_client.get_object(main_bucket, file_name)
            img = Image.open(BytesIO(data.read()))
            img.verify()
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            
            img.save(test_folder + "/" + file_name.split("/")[-1])

         
# Example usage:
minio_url = "YOUR_MINIO_URL"
minio_access_key = "YOUR_ACCESS_KEY"
minio_secret_key = "YOUR_SECRET_KEY"

client = Minio(minio_url, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)
split_dataset_to_minio("data", ['real', 'fake', 'blurred', 'low_quality', 'distorted'], 0.8, client)
