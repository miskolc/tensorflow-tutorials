from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data Sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def download_datasets(local_path, url_path):
    if not os.path.exists(local_path):
        raw = urlopen(url_path).read()
        with open(local_path, "wb") as f:
            f.write(raw)

def load_datasets(filename):
    data_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=filename,
        target_dtype=np.int,
        features_dtype=np.float32)
    return data_set

def input_fn(dataset, num_epochs, shuffle):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(dataset.data)},
        y=np.array(dataset.target),
        num_epochs=num_epochs,
        shuffle=shuffle)
    return input_fn

def main():
    # If the training and test sets are not stored locally, download them.
    print("Downloading {}...".format(IRIS_TRAINING))
    download_datasets(IRIS_TRAINING, IRIS_TRAINING_URL)
    print("Downloading {}...".format(IRIS_TEST))
    download_datasets(IRIS_TEST, IRIS_TEST_URL)
    
    # Load datasets.
    training_set = load_datasets(IRIS_TRAINING)
    test_set = load_datasets(IRIS_TEST)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    print(feature_columns)
    
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    print("Building Deep Neural Network")
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                           hidden_units=[10, 20, 10],
                                           n_classes=3,
                                           model_dir="/tmp/iris_model")
    
    # Define the training inputs
    print("Training inputs")
    train_input_fn = input_fn(training_set, num_epochs=None, shuffle=True)
    
    # Train model.
    print("Train Network")
    classifier.train(input_fn=train_input_fn, steps=2000)
    
    # Define the test inputs
    print("Compute accuracy on test data...")
    test_input_fn = input_fn(test_set, num_epochs=1, shuffle=False)
    
    # Evaluate accuracy
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    
    # Classify two new flower samples.
    print("Classify new flower samples")
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(new_samples)},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    
    print(
        "New Samples, Class Predictions:      {}\n"
        .format(predicted_classes))
    
    print("Done.")

if __name__ == "__main__":
    main()

            
    