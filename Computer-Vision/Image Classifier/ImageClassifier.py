import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
image_directory = "/path/to/directory"
image_size = (256, 256)
batch_size = 32
image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = image_generator.flow_from_directory(
    image_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_data = image_generator.flow_from_directory(
    image_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')
model= Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_data, epochs=10, validation_data=validation_data)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
