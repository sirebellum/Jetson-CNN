import cv2
import subprocess
import sys

# Start video capture before opening rtp connection to avoid keyframe errors
vcap = cv2.VideoCapture("./pi.sdp")
'''
ret, frame = vcap.read()

HOST="pi@192.168.12.4"
# Ports are handled in ~/.ssh/config since we use OpenSSH
COMMAND="/home/pi/gits/RPi-Edge/filestream.sh 10.8.0.3 /home/pi/gits/RPi-Edge/video.mp4"

ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
result = ssh.stdout.readlines()
if result == []:
    error = ssh.stderr.readlines()
    print >>sys.stderr, "ERROR: %s" % error
else:
    print (result)
'''
while(1):
    ret, frame = vcap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)
