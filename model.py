import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers

def build_cnn(input_shape, num_classes):
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    # Conv Block 2
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    # Conv Block 3
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes, activation='softmax'))

    return model
