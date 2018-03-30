from pycocotools import coco
import numpy as np
import skimage.io as io
import cv2
from functions import draw_boxes
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX
def draw_boxes(boxes, frame):
    for box in boxes:
        frame = cv2.rectangle(frame,
                             (int(box[0]), int(box[1])),
                             (int(box[0]+box[2]), int(box[1]+box[3])),
                             (255,255,0),
                             2)

    return frame

def crop_and_warp(image, box): #crop and warp image to box then 32x32
    cropped = image[ int(box[1]):int(box[1]+box[3]),
                     int(box[0]):int(box[0]+box[2]) ]
    warped = cv2.resize(cropped, (32, 32))
    return warped
    
# initialize COCO api for instance annotations
valData='val2017'
trainData='train2017'
dataType = valData
annFile='annotations/instances_{}.json'.format(dataType)

imageDir = 'images/'

coco_handle=coco.COCO(annFile)

# human-readable COCO categories
cats = coco_handle.loadCats(coco_handle.getCatIds())
nms=[cat['name'] for cat in cats]

# get all images containing given categories (nms)
catIds = coco_handle.getCatIds(catNms=nms)
imgIds = coco_handle.getImgIds()

images = list()
labels = list()
for imgId in imgIds:

  #Retrieve image
  img = coco_handle.loadImgs(imgId)[0] #image descriptor
  image_location = imageDir+dataType+'/'+img['file_name']
  image = cv2.imread(image_location) #actual image
  image = np.divide(image, 255.0) #Normalize to [0,1]
  
  #Retrieve bounding boxes
  annIds = coco_handle.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
  anns = coco_handle.loadAnns(annIds) #annotation data
  boxes = list()
  for ann in anns: #get bounding boxes
    boxes.append(ann['bbox'])
  
  #Crop and warp
  for box in boxes:
    warped = crop_and_warp(image, box)
    cv2.imshow('warped', warped)
    cv2.waitKey(0)