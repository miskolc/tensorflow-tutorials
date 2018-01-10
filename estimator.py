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

def main():
    # If the training and test sets are not stored locally, download them.
    print("Downloading {}...".format(IRIS_TRAINING))
    download_datasets(IRIS_TRAINING, IRIS_TRAINING_URL)
    print("Downloading {}...".format(IRIS_TEST))
    download_datasets(IRIS_TEST, IRIS_TEST_URL)

    print("Done.")

if __name__ == "__main__":
    main()

            
    