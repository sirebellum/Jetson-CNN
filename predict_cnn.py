# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_models import CNN_Model
cnn_model = CNN_Model #which model to use
import cv2
from functions import receiverNetwork, draw_boxes, parse_predictions, get_labels, visualize, prune_boxes, crop_and_warp
import time

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Relative path to model")
parser.add_argument("vis", help="Visualizations? y/n")
args = parser.parse_args()
model_path = args.model_name

def main(unused_argv):

  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=model_path)

  labels = get_labels() #maps id to name
  receiver = receiverNetwork(9002) #receive from edgeboxes
  
  total_time = 0
  total_execs = 0
  
  try:
      #get image and boxes from network
      image, boxes = receiver.receiveBoxes()
      while True:
          b_time = time.time() #beginning time
          
          #Create list of all objects, cropped and warped
          objects = list()
          for box in boxes:
              object = crop_and_warp(image, box)
              objects.append(object)
          samples = np.array(objects, dtype=np.float32)
          
          #Input function with all objects in image
          pred_input_fn = tf.estimator.inputs.numpy_input_fn(
              x=samples,
              num_epochs=1,
              shuffle=False)

          predictions = classifier.predict(
              input_fn=pred_input_fn,
              yield_single_examples=False)
              
          classes, scores = parse_predictions(predictions)
          
          #Get rid of 0 objects and merge iou threshold
          boxes, classes, scores = prune_boxes(boxes, 0.7, classes, scores)

          exec_time = time.time()-b_time
          print("Executed in:", exec_time) #execution time
          total_time = total_time + exec_time
          total_execs = total_execs + 1
          
          if args.vis == 'y':
              image = image*255 #Convert to value in [0,255] for vis
              image = image.astype(np.uint8)
              image = visualize(boxes, image, scores, classes, labels)
                  
              cv2.imshow("Image", image)
              cv2.waitKey(10)
              
          #get image and boxes from network
          image, boxes = receiver.receiveBoxes()
              
  except KeyboardInterrupt:
      exit(total_time/total_execs)
  
if __name__ == "__main__":
  tf.app.run()
