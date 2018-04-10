from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import COCO.COCO #coco dataset handle
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_models import CNN_Model
cnn_model = CNN_Model #which model to use

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify model output name")
args = parser.parse_args()

CWD_PATH = os.getcwd()

# Our application logic will be added here

def main(unused_argv):

  # Load training and eval data
  num_objects = 2000 #Number of objects to retrieve at a time
  trainDataset = COCO.COCO.dataset('train')
  train_data, train_labels = ([1], [1])

  # Estimator config to change frequency of ckpt files
  my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20,  # Save checkpoints every 20 seconds.
    keep_checkpoint_max = 5,)       # Retain the 5 most recent checkpoints.
  
  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=CWD_PATH+"/models/"+args.output_name,
    config=my_checkpointing_config )
    
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
      
  # Train the model
  while len(train_data) > 0:
      train_data, train_labels = trainDataset.nextImages(num_objects) #load next set
      train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=256,
        num_epochs=None,
        shuffle=True)
      classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

if __name__ == "__main__":
  tf.app.run()
