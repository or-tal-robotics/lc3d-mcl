#!/usr/bin/env python
import serial
import numpy as np
import rospy
#from sensor_msgs.msg import Imu

ser = serial.Serial('/dev/ttyUSB0',57600,timeout=0.01)

for ii in range(100):
    line = ser.read_until('#',20)
    print line  
    if chr(line[1]) == 'M':
        M = line[5:-2]
        M = M.split(b',')
        try:
            M = np.array(M).astype(np.float)
        except:
            continue
        
        
    if chr(line[1]) == 'G':
        G = line[5:-2]
        G = G.split(b',')
        try:
            G = np.array(G).astype(np.float)
        except:
            continue
        
    if chr(line[1]) == 'A':
        A = line[5:-2]
        A = A.split(b',')
        try:
            A = np.array(A).astype(np.float)
        except:
            continue
        
ser.close() 

#rospy.init_node("imu_bridge", anonymous=True)
#rospy.Publisher("imu", Imu, queue_size=10)