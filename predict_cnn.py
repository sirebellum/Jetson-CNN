# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_models import CNN_Model
cnn_model = CNN_Model #which model to use
import cv2
from functions import receiverNetwork, draw_boxes
from COCO.COCO import crop_and_warp

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Relative path to model")
args = parser.parse_args()
model_path = args.model_name

def main(unused_argv):

  receiver = receiverNetwork(9002)
  while True:
      #get image and boxes from network
      image, boxes = receiver.receiveBoxes()
      cv2.imshow("image", draw_boxes(boxes, image))
      cv2.waitKey(1)
  
      samples = np.array([image],
                         dtype=np.float32)
                     
  if "DISPLAY" in os.environ:
    cv2.imshow("image", image)
    cv2.waitKey(0)
  
  #Input function to take 1 wav and parse into image
  pred_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=samples,
    num_epochs=1,
    shuffle=False)

  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=model_path)

  predictions = classifier.predict(input_fn=pred_input_fn, yield_single_examples=False)
  with open("results.txt", "w") as f: 
    for p in predictions:
      dark = str(p['probabilities'][0][0])
      bright = str(p['probabilities'][0][1])
      print("Bright:", bright, "\nDark:", dark)

      f.truncate() #clear last results
      f.write(dark+"\n") #Dark
      f.write(bright) #Bright
  
if __name__ == "__main__":
  tf.app.run()
