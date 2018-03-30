from pycocotools import coco
import numpy as np
import skimage.io as io
import cv2
from functions import draw_boxes
import matplotlib.pyplot as plt
import math
from multiprocessing import Process, Queue

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
    warped = cv2.resize(cropped, (28, 28))
    return warped

class dataset:

    def __init__(self, data):
    
        if data == 'train': self.dataType = 'train2017'
        elif data == 'test': self.dataType = 'val2017'
        else: exit("invalid datatype (shoudl be test or train)")

        #Multithreading
        self.num_threads = 4
        #Create an output and input queue for each thread
        self.queue = list()
        self.queuein = list()
        for x in range(0, self.num_threads):
            self.queue.append(Queue())
            self.queuein.append(Queue())
        #Initialize Threads
        self.thread = list()
        for x in range(0, self.num_threads):
            self.thread.append(Process(target=parseImage,
                               args=(self.queuein[x],
                                     self.queue[x],),
                               daemon = True))
            self.thread[x].start()
        
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
        
        print("loading images into memory...")
        images = list()
        labels = list()
        for x in range(self.numImages, len(self.imgIds)):
          for thread in range(0, self.num_threads):
            #Retrieve image location
            img = self.coco_handle.loadImgs(self.imgIds[x+thread])[0] #image descriptor
            image_location = self.imageDir+self.dataType+'/'+img['file_name']
            #Retrieve annotations
            annIds = self.coco_handle.getAnnIds(imgIds=self.imgIds[x+thread],
                                                catIds=self.catIds,
                                                iscrowd=None)
            anns = self.coco_handle.loadAnns(annIds) #annotation data
            #Execute thread
            self.queuein[thread].put(image_location)
            self.queuein[thread].put(anns)
          
          for queue in self.queue:
            images = images + queue.get()
            labels = labels + queue.get()
          
          self.numImages = self.numImages + self.num_threads
          
          if len(images) >= numObjects and numObjects != -1: #Get all objects if numObjects -1
            print(len(images), "objects warped")
            break
        
        
        print(len(self.imgIds) - self.numImages, "images left.")
          
        return np.asarray(images), np.asarray(labels, dtype=np.int32)

def parseImage(qin, q):
  while True:
    file = qin.get()
    annotations = qin.get()
    image = cv2.imread(file) #actual image
    image = image.astype(np.float32)
    image = np.divide(image, 255.0) #Normalize to [0,1]
    
    labels = list()
    images = list()
    for ann in annotations: #get bounding boxes
      labels.append(labeled(ann['category_id']))
      images.append(crop_and_warp(image, ann['bbox']))
      
    q.put(images)
    q.put(labels)
        

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
