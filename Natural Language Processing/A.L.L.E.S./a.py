import tensorflow as tf

model_path = '/path/to/directory'

model = tf.keras.models.load_model(model_path)

model.summary()
