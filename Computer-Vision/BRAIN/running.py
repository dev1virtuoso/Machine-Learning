import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import MNN

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(14,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(14, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train = np.random.rand(1000, 14)
y_train = np.random.rand(1000, 14)

model.fit(X_train, y_train, epochs=10, batch_size=32)

converter = MNN.converter(model)
converter.convert('running_model.mnn')

interpreter = MNN.Interpreter('running_model.mnn')
session = interpreter.createSession()

X_test = np.random.rand(10, 14)

input_tensor = interpreter.getSessionInput(session)
input_tensor.copyFrom(X_test)
interpreter.runSession(session)
output_tensor = interpreter.getSessionOutput(session)
y_pred = output_tensor.copyToNumpy()

def plot_skeleton(data, title):
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), ]
