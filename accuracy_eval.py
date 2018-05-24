# Imports
import numpy as np
import os
import argparse
import tensorflow as tf
from cnn_models import CNN_Model
cnn_model = CNN_Model #which model to use
import cv2
from functions import receiverNetwork, draw_boxes, parse_predictions, get_labels, visualize, write_file
from COCO.COCO import crop_and_warp
from COCO.COCOlite import dataset
import visualization_utils as vis_utils #tensorflow provided vis tools
import subprocess
import time

#DEBUG, INFO, WARN, ERROR, or FATAL
tf.logging.set_verbosity(tf.logging.WARN)

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Relative path to model")
parser.add_argument("vis", help="Visualizations? y/n")
args = parser.parse_args()
model_path = args.model_name


#EDGEBOXES setup
modelfile = "./model.yml.gz"
print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
boxGenerator = cv2.ximgproc.createEdgeBoxes(maxBoxes = 1000,
                                            alpha = 0.65,
                                            beta = 0.75,
                                            minScore = 0.03)

COCO = dataset() #Retrieve images from COCO

def main(unused_argv):

  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model,
    model_dir=model_path)
    
  labels = get_labels() #maps id to name

  
  total_time = 0
  total_execs = 0
  
  try:
    #GroundTruth
    image , gt_classes, gt_boxes, filename = COCO.nextImage()
    while image is not None:
      #EdgeBoxes
      edgearray = edgeGenerator.detectEdges(image)
      orientationarray = edgeGenerator.computeOrientation(edgearray)
      suppressed_edgearray = edgeGenerator.edgesNms(edgearray, orientationarray)
      boxes = boxGenerator.getBoundingBoxes(suppressed_edgearray, orientationarray)

      b_time = time.time() #beginning time
      
      #Create list of all objects, cropped and warped
      objects = list()
      for box in boxes:
          object = crop_and_warp(image, box)
          objects.append(object)
      samples = np.array(objects, dtype=np.float32)
      
      if len(boxes) > 0: #skip images with no boxes
          #Input function with all objects in image
          pred_input_fn = tf.estimator.inputs.numpy_input_fn(
              x=samples,
              num_epochs=1,
              shuffle=False)

          predictions = classifier.predict(
              input_fn=pred_input_fn,
              yield_single_examples=False)

          classes, scores = parse_predictions(predictions) #predictions is a weird object
          
          exec_time = time.time()-b_time
          print("Executed in:", exec_time) #execution time
          total_time = total_time + exec_time
          total_execs = total_execs + 1
          
          mAP_paths = ["./mAP/", filename] #Path to mAP https://github.com/Cartucho/mAP
          write_file(gt_classes, gt_boxes, mAP_paths, None, labels) #write gt files
          write_file(classes, boxes, mAP_paths, scores, labels) #write predicted files
          
          if args.vis == 'y':
              image = image*255 #Convert to value in [0,255] for vis
              image = image.astype(np.uint8)
              image_COCO = visualize(gt_boxes, image, None, gt_classes, labels)
              image = visualize(boxes, image, scores, classes, labels)

              cv2.imshow("COCO", image_COCO)
              cv2.imshow("image", image)
              cv2.waitKey(1000)
              
      #GroundTruth
      image , gt_classes, gt_boxes, filename = COCO.nextImage()
  
  except KeyboardInterrupt:
      exit(total_time/total_execs)
  
if __name__ == "__main__":
  tf.app.run()
