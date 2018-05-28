import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import socket
import pickle
import numpy as np
import math

def prune_boxes(boxes, threshold, classes):
    #Create dictionary of box indices corresponding to classes
    boxes_with_class = {}
    i = 0
    for clas in classes:
        if clas not in boxes_with_class.keys() and clas != 0: #ignore background objects
            boxes_with_class[clas] = []
        if clas != 0:
            boxes_with_class[clas].append(i) #add index
        i = i + 1

    #Remove overlapping boxes of same class
    good_boxes = list()
    good_classes = list()
    for id in boxes_with_class:
        class_boxes = boxes[boxes_with_class[id]]
        suppressed_boxes = non_max_suppression_fast(class_boxes, threshold)
        for x in range(0, len(suppressed_boxes)):
            good_classes.append(id)
            good_boxes.append(suppressed_boxes[x])

    return np.asarray(good_boxes, dtype=np.uint16), np.asarray(good_classes, dtype=np.uint8)

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]+boxes[:,0]
    y2 = boxes[:,3]+boxes[:,1]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    good_boxes = boxes[pick].astype("int")
    good_boxes[:,2] = good_boxes[:,2]-good_boxes[:,0]
    good_boxes[:,3] = good_boxes[:,3]-good_boxes[:,1]
    return good_boxes

def crop_and_warp(image, box): #crop and warp image to box then 32x32
    cropped = image[ math.floor(box[1]):math.ceil(box[1]+box[3]),
                     math.floor(box[0]):math.ceil(box[0]+box[2]) ]
    warped = cv2.resize(cropped, (225, 225))
    #if not isinstance(warped[0][0][0], float): #freaks out if it's a float
    #  warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return warped

def IoU(boxA, boxB):
    #Convert xy coordinates
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def write_file(clazzes, boxes, paths, scores, labels):

    filename = paths[1].strip(".jpg")+".txt" #write txt not jpg
    mAP_path = paths[0]
    if scores is None:
        path = mAP_path+"ground-truth/"+filename
    else:
        path = mAP_path+"predicted/"+filename
    
    with open(path, 'w') as f:
        i = 0
        if scores is None: #If a ground truth file
            for clazz in clazzes:
                left = int(boxes[i][0])
                top = int(boxes[i][1])
                right = int(boxes[i][0])+int(boxes[i][2])
                bottom = int(boxes[i][1])+int(boxes[i][3])
                name = labels[clazz]['name'].replace(' ', '-') #replace for mAP
                string = "{} {} {} {} {}\n".format(name, left, top, right, bottom)
                f.write(string)
                i = i + 1
        elif scores[i] >= 0.0: #If score is above threshold
            for clazz in clazzes:
                score = scores[i]
                left = int(boxes[i][0])
                top = int(boxes[i][1])
                right = int(boxes[i][0])+int(boxes[i][2])
                bottom = int(boxes[i][1])+int(boxes[i][3])
                name = labels[clazz]['name'].replace(' ', '-') #replace for mAP
                string = "{} {} {} {} {} {}\n".format(name, score, left, top, right, bottom)
                f.write(string)
                i = i + 1
        else:
            i = i + 1

def draw_boxes(boxes, frame):
    frame2 = frame.copy()

    for box in boxes:
        frame2 = cv2.rectangle(frame2,
                               (int(box[0]), int(box[1])),
                               (int(box[0]+box[2]), int(box[1]+box[3])),
                               (255,255,0),
                               2)

    return frame2

def draw_labels(labels, classes, frame, boxes):
    frame2 = frame.copy()
    
    i = 0
    for id in classes:
        frame2 = cv2.putText(frame2,
                   labels[id]['name'],
                   (int(boxes[i][0]),int(boxes[i][1]-5)),
                   font,
                   0.5,
                   (255,255,255),
                   1,
                   cv2.LINE_AA)
        i = i + 1

    return frame2
    
def visualize(boxes, frame, scores, classes, labels, threshold=0.9):

    good_boxes = list()
    good_classes = list()
    i = 0
    for box in boxes:
       if scores is None: #draw all if scores is None
         good_boxes.append(box)
         good_classes.append(classes[i])
       elif scores[i] >= threshold:
         good_boxes.append(box)
         good_classes.append(classes[i])
       i = i + 1
    
    image = draw_boxes(good_boxes, frame)
    image = draw_labels(labels, good_classes, image, good_boxes)
    return image
    
def parse_predictions(predictions):

    scores = list()
    classes = list()
    for p in predictions:
        class_batch = p["classes"]
        classes.append(class_batch.tolist())
        
        probs = p["probabilities"] #includes scores for other classes
        i = 0
        for prob in probs: #retrieve score for decided class
            class_id = class_batch[i]
            scores.append(prob[class_id])
            i = i + 1

    #Flatten
    classes = [item for sublist in classes for item in sublist]
    #Convert to numpy
    classes = np.asarray(classes, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float16)
    
    return classes, scores
    
def get_labels():

    labels = list()
    with open("COCO/labels.txt", 'r') as f:
      line = f.readline()
      while line:
        id, name = line.split(",")
        name = name.strip("\n")
        labels.append({"id":int(id), "name":name})
        line = f.readline()
    
    category_index = {}
    for cat in labels:
      category_index[cat['id']] = cat
    
    return category_index

class receiverNetwork:

    def __init__(self, port):
    
        self.UDP_IP = "127.0.0.1" #IP of network interface to be used
        self.sock = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP
        self.UDP_PORT = port
        self.sock.bind((self.UDP_IP, self.UDP_PORT))

    #Transmit boxes and frame
    def receiveBoxes(self):

        buffer = self.sock.recv(65525)
        
        data = pickle.loads(buffer) #unpickle list of img and boxes
        
        image_compressed = data[0] #get img
        image_numpy = np.fromstring(image_compressed, dtype=np.uint8)
        image = cv2.imdecode(image_numpy, cv2.IMREAD_UNCHANGED) #uncompress
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #tensorflow expectations
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        boxes = pickle.loads(data[1]) #unpickle boxes
        
        return image, boxes
