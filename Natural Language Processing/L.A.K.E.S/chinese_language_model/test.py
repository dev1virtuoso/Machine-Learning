import tensorflow as tf

model = tf.keras.models.load_model('/path/to/directory')

loss, accuracy = model.evaluate(input_sequences, target_sequences)
print('Loss:', loss)
print('Accuracy:', accuracy)
