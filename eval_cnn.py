from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import COCO.COCO
import numpy as np
import os
import argparse
import inotify.adapters
import tensorflow as tf
from cnn_models import cnn_model_5x1
cnn_model = cnn_model_5x1 #which model to use

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
args = parser.parse_args()

CWD_PATH = os.getcwd()
model_path = CWD_PATH+"/models/"+args.output_name

#Inotify setup
file_watch = inotify.adapters.Inotify()
file_watch.add_watch(model_path)

def main(unused_argv):

  # Load training and eval data
  trainDataset = COCO.COCO.dataset('test')
  eval_data, eval_labels = trainDataset.nextImages(-1)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model, model_dir=model_path)
    
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

  for event in file_watch.event_gen(yield_nones=False): #Evaluate for every new file
    # Evaluate the model and print results
    (_, type_names, path, filename) = event
    if type_names[0] is 'IN_MOVED_TO' and 'checkpoint' in filename and 'tmp' not in filename:
      eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
  
if __name__ == "__main__":
  tf.app.run()
