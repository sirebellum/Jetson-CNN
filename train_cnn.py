from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_models import CNN_Model, parse_record, train_input_fn
cnn_model = CNN_Model #which model to use

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
args = parser.parse_args()

CWD_PATH = os.getcwd()

def main(unused_argv):

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.45

  # Estimator config to change frequency of ckpt files
  estimator_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20,  # Save checkpoints every 20 seconds.
    keep_checkpoint_max = 5,
    session_config=config)       # Retain the 5 most recent checkpoints.
  
  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=CWD_PATH+"/models/"+args.output_name,
    config=estimator_config )
    
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
      
  # Train the model
  classifier.train(
        input_fn=train_input_fn,
        steps=200000,
        hooks=[logging_hook])

if __name__ == "__main__":
  tf.app.run()
