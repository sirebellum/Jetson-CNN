import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import socket
import pickle
import numpy as np

def draw_boxes(boxes, frame):

    frame2 = frame.copy()

    for box in boxes:
        frame2 = cv2.rectangle(frame2,
                               (int(box[0]), int(box[1])),
                               (int(box[0]+box[2]), int(box[1]+box[3])),
                               (255,255,0),
                               2)

    return frame2

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #tensorflow expectations
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        boxes = pickle.loads(data[1]) #unpickle boxes
        
        return image, boxes