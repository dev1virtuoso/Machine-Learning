from tensorflow import keras

model = keras.models.load_model('/path/to/directory')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
