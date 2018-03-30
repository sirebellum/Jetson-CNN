from pycocotools import coco
import numpy as np
import skimage.io as io
import cv2
from functions import draw_boxes
import matplotlib.pyplot as plt
import math

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
    cropped = image[ math.floor(box[1]):math.ceil(box[1]+box[3]),
                     math.floor(box[0]):math.ceil(box[0]+box[2]) ]
    warped = cv2.resize(cropped, (32, 32))
    return warped

class dataset:

    def __init__(self, data):
    
        if data == 'train': self.dataType = 'train2017'
        elif data == 'test': self.dataType = 'val2017'
        else: exit("invalid datatype (shoudl be test or train)")
    
        # initialize COCO api for instance annotations
        annFile='COCO/annotations/instances_{}.json'.format(self.dataType)
        self.imageDir = 'COCO/images/'

        self.coco_handle=coco.COCO(annFile)

        # human-readable COCO categories
        cats = self.coco_handle.loadCats(self.coco_handle.getCatIds())
        nms=[cat['name'] for cat in cats]

        # get all images containing given categories (nms)
        self.catIds = self.coco_handle.getCatIds(catNms=nms)
        self.imgIds = self.coco_handle.getImgIds()
        
        self.numImages = 0 #number of processed images
        
        print(len(self.imgIds), "total images in", data, "set.")
        
    def nextImages(self, numObjects): #return aprox. numObjects warped and cropped objects
        
        images = list()
        labels = list()
        for x in range(self.numImages, len(self.imgIds)):

          #Retrieve image
          img = self.coco_handle.loadImgs(self.imgIds[x])[0] #image descriptor
          image_location = self.imageDir+self.dataType+'/'+img['file_name']
          image = cv2.imread(image_location) #actual image
          image = np.divide(image, 255.0) #Normalize to [0,1]
          
          #Retrieve bounding boxes and warp images
          annIds = self.coco_handle.getAnnIds(imgIds=self.imgIds[x],
                                              catIds=self.catIds,
                                              iscrowd=None)
          anns = self.coco_handle.loadAnns(annIds) #annotation data
          boxes = list()
          for ann in anns: #get bounding boxes
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            images.append(crop_and_warp(image, ann['bbox']))
            
          self.numImages = self.numImages + 1
          
          if len(images) >= numObjects:
            print(len(images), "objects warped")
            break
          
        print(len(self.imgIds) - self.numImages, "images left.")
          
        return images, labels
