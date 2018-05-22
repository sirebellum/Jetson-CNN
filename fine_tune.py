import tensorflow as tf
import argparse
from tensorflow.python.tools import inspect_checkpoint as chkp

tf.logging.set_verbosity(tf.logging.WARN)
#DEBUG, INFO, WARN, ERROR, or FATAL

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Relative path to model checkpoint")
args = parser.parse_args()

tf.reset_default_graph()
chkp.print_tensors_in_checkpoint_file(args.model_dir+"/model.ckpt-422680",
                                      tensor_name='',
                                      all_tensors=False,
                                      all_tensor_names=True)

exit()
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, args.model_dir+"/model.ckpt")
  print("Model restored.")
