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

def parser(serialized_example):
  """Parses a single tf.Example into image and label tensors."""
  features = {'image/encoded': tf.FixedLenFeature([], tf.string),
             'image/format':  tf.FixedLenFeature([], tf.string),
             'image/label':   tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(serialized_example, features)
  
  #print("JPG:", features['image/encoded'])
  image = tf.image.decode_jpeg(features['image/encoded'], channels=0)
  image = tf.cast(image, tf.float32) #Change from uint8 to float for compatibility with reshape
  #print("image:", image)
  image = tf.reshape(image, [225, 225, 3])
  
  label = tf.cast(features['image/label'], tf.int32)
  
  return (image, label)

# Define the input function for training
def train_input_fn():

  # Keep list of filenames, so you can input directory of tfrecords easily
  train_filenames = ["COCO/train.record"]
  test_filenames = ["COCO/test.record"]
  batch_size=256

  # Import data
  dataset = tf.data.TFRecordDataset(train_filenames)

  # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(parser)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  #print("Dataset:", dataset.output_shapes, ":::", dataset.output_types)
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  #print("Iterator:", features)

  return (features, labels)

# Our application logic will be added here

def main(unused_argv):

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
  classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

if __name__ == "__main__":
  tf.app.run()
