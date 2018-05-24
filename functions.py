import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import socket
import pickle
import numpy as np

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
                right = int(boxes[i][2])
                bottom = int(boxes[i][3])
                name = labels[clazz]['name']
                string = "{} {} {} {} {}\n".format(name, left, top, right, bottom)
                f.write(string)
                i = i + 1
        else: #If a predicted file
            for clazz in clazzes:
                score = scores[i]
                left = boxes[i][0]
                top = boxes[i][1]
                right = boxes[i][2]
                bottom = boxes[i][3]
                name = labels[clazz]['name']
                string = "{} {} {} {} {} {}\n".format(name, score, left, top, right, bottom)
                f.write(string)
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
    classes = np.asarray(classes)
    scores = np.asarray(scores)
    
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
