from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import urllib
import numpy as np
import tensorflow as tf
import pandas as pd

IRIS_TRAINING = "/path/to/directory"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
IRIS_TEST = "/path/to/directory"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)
    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)

    training_data = pd.read_csv(IRIS_TRAINING, header=0)
    training_features = training_data.iloc[:, :4].values
    training_labels = training_data.iloc[:, 4].values

    test_data = pd.read_csv(IRIS_TEST, header=0)
    test_features = test_data.iloc[:, :4].values
    test_labels = test_data.iloc[:, 4].values

    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_features)},
        y=np.array(training_labels),
        num_epochs=None,
        shuffle=True)

    classifier.train(input_fn=train_input_fn, steps=2000)

    test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_features)},
        y=np.array(test_labels),
        num_epochs=1,
        shuffle=False)

    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("Test accuracy: {0:f}".format(accuracy_score))

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
