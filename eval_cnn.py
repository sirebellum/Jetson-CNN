from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import os
import argparse
import inotify.adapters
import tensorflow as tf
from cnn_models import CNN_Model, parse_record, eval_input_fn
cnn_model = CNN_Model #which model to use

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("output_name", help="Specify path to model")
parser.add_argument("eval_imm", default=0, help="Evaluate the most recent checkpoint")
args = parser.parse_args()

CWD_PATH = os.getcwd()
if "models" and "/" not in args.output_name:
  model_path = CWD_PATH+"/models/"+args.output_name

#Inotify setup
file_watch = inotify.adapters.Inotify()
file_watch.add_watch(model_path)

def main(unused_argv):

  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model, model_dir=model_path)

  for event in file_watch.event_gen(yield_nones=False): #Evaluate for every new file
    # Evaluate the model and print results
    (_, type_names, path, filename) = event
    new_ckpt = type_names[0] is 'IN_MOVED_TO' and 'checkpoint' in filename and 'tmp' not in filename
    cli_arg = args.eval_imm
    if new_ckpt or cli_arg:
      print("Evaluating...")
      eval_results = classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)
      cli_arg = 0 #don't run immediately again
  
if __name__ == "__main__":
  tf.app.run()
