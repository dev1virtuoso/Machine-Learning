# Copyright (c) 2024 Carson. All rights reserved.

import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('/path/to/directory')

# Evaluate the model
loss, accuracy = model.evaluate(input_sequences, target_sequences)
print('Loss:', loss)
print('Accuracy:', accuracy)
