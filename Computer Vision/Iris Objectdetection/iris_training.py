from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import urllib
import numpy as np
import tensorflow as tf
import pandas as pd

# Dataset file paths
IRIS_TRAINING = "/path/to/directory"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
IRIS_TEST = "/path/to/directory"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # Download the training and test datasets if they don't exist
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)

    # Load the datasets
    training_data = pd.read_csv(IRIS_TRAINING, header=0)
    training_features = training_data.iloc[:, :4].values
    training_labels = training_data.iloc[:, 4].values

    test_data = pd.read_csv(IRIS_TEST, header=0)
    test_features = test_data.iloc[:, :4].values
    test_labels = test_data.iloc[:, 4].values

    # Specify that all features are numeric
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build a DNN model with 3 hidden layers of 10, 20, and 10 units each
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

    # Define the training input function
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_features)},
        y=np.array(training_labels),
        num_epochs=None,
        shuffle=True)

    # Train the model
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Define the test input function
    test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_features)},
        y=np.array(test_labels),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("Test accuracy: {0:f}".format(accuracy_score))

    # Perform classification predictions on two new flower samples
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    print("Predicted classes for new samples: {}".format(predicted_classes))
    
    saved_model_path = "/path/to/directory"
    tf.saved_model.save(classifier, saved_model_path)

if __name__ == "__main__":
    main()
