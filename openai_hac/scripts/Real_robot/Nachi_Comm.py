import socket
import time
import math
import cv2 as cv
import numpy as np
import serial
from threading import Thread
# define constants for socket commnuncation
class Socket_comm():
    def __init__(self):
        self.PORT_NUM = 48952
        self.SEND_DATA_SIZE = 8
        self.SEND_BUFFER_LEN = self.SEND_DATA_SIZE * 6
        self.REC_DATA_SIZE = 12
        self.REC_DATA_NUM = 7
        self.REC_IO_DATA_SIZE = 3
        self.REC_BUFFER_LEN = self.REC_DATA_SIZE * 6 + self.REC_IO_DATA_SIZE + self.REC_DATA_NUM



        self.MACHINE_ABS_LINEAR = 1          # MOVE BY ABS COORDINATE VALUE RESPECT MACHINE COORDINATE FRAME USING LINEAR INTERPOLATION
        self.MACHINE_ABS_JOINT = 2           # ...
        self.MACHINE_REALATIVE_LINEAR = 3    # ...
        self.MACHINE_REALATIVE_JOINT = 4     # MOVE BY REALATIVE COORDINATE VALUE RESPECT MACHINE COORDINATE FRAME USING JOINT INTERPOLATION

        self.JOINT_ABS_LINEAR = 5            # MOVE BY ABS COORDINATE VALUE RESPECT JOINT COORDINATE FRAME USING LINEAR INTERPOLATION
        self.JOINT_ABS_JOINT = 6             # ...
        self.JOINT_REALATIVE_LINEAR = 7      # ...
        self.JOINT_REALATIVE_JOINT = 8       # MOVE BY REALATIVE COORDINATE VALUE RESPECT JOINT COORDINATE FRAME USING JOINT INTERPOLATION

        self.OPEN_COMPRESSED_AIR = 9
        self.ClOSE_COMPRESSED_AIR = 10


        # define a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = (('192.168.1.1', self.PORT_NUM))        # Controller IP address

    def socket_initalize(self):
        # server_address = (('localhost', PORT_NUM))
        print('Connecting to {} port {}'.format(*self.server_address))
        # Connect the socket to the port where the server is listening

        self.sock.connect(self.server_address)
    def socket_close(self):
        self.sock.close()

    def tool_coordinate(self):
        
        #Get tool_coordinate
        M = "P"
        M = bytes(M, 'utf-8')
        # M = M.decode('utf-8')
        self.sock.sendall(M)             #send signal to get data
        data = self.sock.recv(1024)      #buffer size
        data = data.decode("utf-8")
        data = data.split(",")
        print("-----------------------")
        print("Current Tool Position")
        print("-----------------------")

        print('X    :  ', data[0])
        print('Y    :  ', data[1])
        print('Z    :  ', data[2])
        print('Roll :  ', data[3])
        print('Pitch:  ', data[4])
        print('Yaw  :  ', data[5])

    def joint_coordinate(self):
        M = "J"
        M = bytes(M, 'utf-8')
        # M = M.decode('utf-8')
        self.sock.sendall(M)
        data = self.sock.recv(1024)
        data = data.decode("utf-8")
        data = data.split(",")
        print("-----------------------")
        print("Current Joint Position")
        print("-----------------------")

        print('Joint 1 :  ', data[0])
        print('Joint 2 :  ', data[1])
        print('Joint 3 :  ', data[2])
        print('Joint 4 :  ', data[3])
        print('Joint 5 :  ', data[4])
        print('Joint 6 :  ', data[5])

    def move_robot(self, move_coord, move_mode):        # Move I-K (3 mode)

        # wait for machine 'REA'DY status
        M = bytes("A", 'utf-8')
        signal = "busy"
        while True:
            self.sock.sendall(M)
            signal = self.sock.recv(3)
            if signal == b'REA':
                break

        # data preparation
        x = "{0:8.2f}".format(move_coord[0])
        y = "{0:8.2f}".format(move_coord[1])
        z = "{0:8.2f}".format(move_coord[2])
        r = "{0:8.2f}".format(move_coord[3])
        p = "{0:8.2f}".format(move_coord[4])
        ya = "{0:8.2f}".format(move_coord[5])
        mode = "{:0>3d}".format(move_mode)
        
        # binding data and converting
        message = x + y + z + r + p + ya + mode
        message = bytes(message, 'utf-8')

        # send
        self.sock.sendall(message)
        # wait for machine 'FIN'ISH status
        
        M = bytes("A", 'utf-8')
        signal = "busy"
        
        while True:
            signal = self.sock.recv(3)
            if signal == b'FIN':
                break

    def IO_robot(self, move_mode):

        # wait for machine 'REA'DY status
        M = bytes("A", 'utf-8')
        signal = "busy"
        while True:
            self.sock.sendall(M)
            signal = self.sock.recv(3)
            if signal == b'REA':
                break


        # data preparation
        x = "{0:8.2f}".format(0)
        y = "{0:8.2f}".format(0)
        z = "{0:8.2f}".format(0)
        r = "{0:8.2f}".format(0)
        p = "{0:8.2f}".format(0)
        ya = "{0:8.2f}".format(0)
        mode = "{:0>3d}".format(move_mode)

        # binding data and converting
        message = x + y + z + r + p + ya + mode
        message = bytes(message, 'utf-8')
        # send
        self.sock.sendall(message)
        
        # wait for machine 'FIN'ISH status
        M = bytes("A", 'utf-8')
        signal = "busy"
        while True:
            signal = self.sock.recv(3)
            if signal == b'FIN':
                break


    def moveposition(self):
        print("move to home position")
        self.move_robot(home_pos, self.MACHINE_ABS_LINEAR)

home_pos = [0, 0, 0, 0, -90, 0]
Robot = Socket_comm()
Robot.socket_initalize()
Robot.joint_coordinate()
Robot.tool_coordinate()
Robot.moveposition()


