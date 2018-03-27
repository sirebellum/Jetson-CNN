from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_mnist import cnn_model_fn

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
args = parser.parse_args()

CWD_PATH = os.getcwd()

def main(unused_argv):

  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=CWD_PATH+"/models/"+args.output_name)

  while True: #Evaluate forever
     # Evaluate the model and print results
      eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
      eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
  
if __name__ == "__main__":
  tf.app.run()