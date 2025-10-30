import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, y_train, x_test, y_test

def preprocess(x):
    # Normalize pixel values to [0,1]
    x = x.astype('float32') / 255.0
    return x

def get_generators(x_train, y_train, x_val, y_val, batch_size=64, augment=True):
    if augment:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
    else:
        train_datagen = ImageDataGenerator()
    
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_gen = val_datagen.flow(x_val, y_val, batch_size=batch_size)

    return train_gen, val_gen
