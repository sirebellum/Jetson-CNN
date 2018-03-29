import cv2
import subprocess
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("clientip", help="IP address of client running stream.py")
args = parser.parse_args()

HOST="pi@192.168.12.4"
# Ports are handled in ~/.ssh/config since we use OpenSSH
COMMAND="/home/pi/gits/RPi-Edge/filestream.sh " + args.clientip + " /home/pi/gits/RPi-Edge/video.mp4 y > /dev/null 2>&1 &"

print ("sshing...")
ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                       shell=False)
#result = ssh.stdout.readlines()
print ("DONE SSHING")

#filestream.sh above is coded to sleep for a second before executing.
#This is necessary because below command failes if started before stream starts
vcap = cv2.VideoCapture("./pi.sdp")
while(1):
    ret, frame = vcap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)
