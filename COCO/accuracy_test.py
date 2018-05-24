from pycocotools import coco
import numpy as np
import cv2
import math
import tensorflow as tf

class dataset:

    def __init__(self):
    
        self.dataType = 'val2017'
        
        # initialize COCO api for instance annotations
        annFile='./COCO/annotations/instances_{}.json'.format(self.dataType)
        self.imageDir = './COCO/images/'

        self.coco_handle=coco.COCO(annFile)

        # human-readable COCO categories
        cats = self.coco_handle.loadCats(self.coco_handle.getCatIds())
        nms=[cat['name'] for cat in cats]

        # get all images containing given categories (nms)
        self.catIds = self.coco_handle.getCatIds(catNms=nms)
        self.imgIds = self.coco_handle.getImgIds()
        self.totalImages = len(self.imgIds)
        self.numImages = 0 #number of processed images
        
        print(len(self.imgIds), "total images in", self.dataType, "set.")

    def nextImage(self): #return next image
        
        if self.numImages >= self.totalImages:
            print("No more images!")
            return None, None, None
        
        #Retrieve image location
        img = self.coco_handle.loadImgs(self.imgIds[self.numImages])[0] #image descriptor
        image_location = self.imageDir+self.dataType+'/'+img['file_name']
        #Retrieve annotations
        annIds = self.coco_handle.getAnnIds(imgIds=self.imgIds[self.numImages],
                                            catIds=self.catIds,
                                            iscrowd=None)
        anns = self.coco_handle.loadAnns(annIds) #annotation data
        image, labels, boxes = parseAnnotation(image_location, anns)
        
        image = image.astype(np.float32)
        image = np.divide(image, 255.0)
        
        self.numImages = self.numImages + 1
          
        return image, labels, boxes

def parseAnnotation(file, annotations):

    image = cv2.imread(file) #actual image
    #cv2.imshow("image", image)
    #cv2.waitKey(1)
    if image is None: exit("No image!")
    
    boxes = list()
    labels = list()
    for ann in annotations: #get bounding boxes
      boxes.append(ann['bbox'])
      labels.append(labeled(ann['category_id']))
        
    return image, labels, boxes
        

def labeled(id): #normalize labels to fit within 80
    if id == 81: return 12
    elif id == 82: return 26
    elif id == 84: return 30
    elif id == 85: return 45
    elif id == 86: return 66
    elif id == 87: return 68
    elif id == 88: return 69
    elif id == 89: return 71
    elif id == 90: return 29
    else: return id
