import cv2

vcap = cv2.VideoCapture("./pi.sdp")
while(1):
    ret, frame = vcap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)
