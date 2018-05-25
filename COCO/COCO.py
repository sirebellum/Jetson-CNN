from pycocotools import coco
import numpy as np
import cv2
import math
from multiprocessing import Process, Queue
import tensorflow as tf
from functions import crop_and_warp, IoU
import random

class dataset:

    def __init__(self, data):
    
        if data == 'train': self.dataType = 'train2017'
        elif data == 'test': self.dataType = 'val2017'
        else: exit("invalid datatype (shoudl be test or train)")

        #Multithreading
        self.num_threads = 7
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
        annFile='./annotations/instances_{}.json'.format(self.dataType)
        self.imageDir = './images/'

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
        for x in range(self.numImages, len(self.imgIds), self.num_threads):
          for thread in range(0, self.num_threads):

            if x+thread >= len(self.imgIds): #if no more images,
              for i in range(0, thread-1):  #purge queues and break
                images = images + self.queue[i].get()
                labels = labels + self.queue[i].get()
              print("No more images!")
              print(len(images), "objects warped")
              self.numImages = self.numImages + thread - 1
              return images, np.asarray(labels, dtype=np.int64)
            
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
            image_queue = queue.get()
            images = images + image_queue
            label_queue = queue.get()
            labels = labels + label_queue
          
          self.numImages = self.numImages + self.num_threads
          
          if len(images) >= numObjects and numObjects != -1: #Get all objects if numObjects -1
            print(len(images), "objects warped")
            break
        
        print(len(self.imgIds) - self.numImages, "images left.")
          
        return images, np.asarray(labels, dtype=np.uint16)

def parseImage(qin, q):
  while True:
    file = qin.get()
    annotations = qin.get()
    image = cv2.imread(file) #actual image
    height, width, channels = image.shape
    #cv2.imshow("image", image)
    #cv2.waitKey(1)
    if image is None: exit("No image!")
    
    labels = list()
    images = list()
    boxes = list()
    for ann in annotations: #get bounding boxes
      boxes.append(ann['bbox'])
      object = crop_and_warp(image, ann['bbox'])
      
      if object.any(): #If not all zeroes
        #cv2.imshow("object", object)
        #cv2.waitKey(1)
        encoded_object = cv2.imencode(".jpg", object)[1].tostring()
        images.append(encoded_object)
        labels.append(labeled(ann['category_id']))
        
      #else:
        #cv2.imshow("suckage", object)
        #cv2.waitKey(1)
        
    #Generate boxes for random background objects
    randos = np.empty((5, 4), dtype=np.uint16)
    for rando in randos:
      overlapping = True
      while overlapping:
        rando[0] = random.randint(0, width-1)
        rando[1] = random.randint(0, height-1)
        rando[2] = random.randint(1, width-rando[0])
        rando[3] = random.randint(1, height-rando[1])
        overlapping = False
        for box in boxes: #check total overlap with actual objects
            iou = IoU(rando, box)
            if iou > 0.01:
                overlapping = True
                break
            
    #Add background objects to image list
    for rando in randos:
        object = crop_and_warp(image, rando)
        encoded_object = cv2.imencode(".jpg", object)[1].tostring()
        images.append(encoded_object)
        labels.append(0)
        cv2.imshow("image", object)
        cv2.waitKey(1000)

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
