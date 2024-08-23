import os
from os.path import isfile, isdir, join, expanduser
import cv2
import numpy as np
from datetime import datetime
import time
from fps import FPS
import csv
import urllib.request
from threading import Thread
from matplotlib import pyplot as plt
import statistics
import matplotlib.animation as anim
import matplotlib.figure as mpl_fig
import serial
import threading
import getch
import random
import zmq
from PIL import Image
import h5py
import queue

#importing required modules
from pygame import mixer
#mixer function call
mixer.init()
#storing the sound after accessing it
plays=mixer.Sound("beep-10.wav")
plays_question=mixer.Sound("question.wav")
plays_end=mixer.Sound("end.wav")

# before copy/paste this line in a terminal
#sudo chmod a+rw /dev/ttyACM0

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)


save = True
# Saving structure
if save==True:
    home = expanduser("~")
    print('Enter the number of the participant:')
    part = getch.getch()
    DIR = f"{home}/Data/image_test/{part}/"
    os.makedirs(DIR, exist_ok=True) # Succeeds if folder already exists
    print(f"[INFO][SAVING] Made working directory: {DIR}")

## Setup the serial communication
serialPort = serial.Serial(port="/dev/ttyACM0", baudrate=115200, bytesize=8, timeout=2)

time.sleep(2)
#serialPort.stopbits = 2

serialString = ""  # Used to hold data coming over UART
framerate1 = 50 #desired fps for force and position data acquisition

# Calibration
print('Start Calibrating, do not touch the object!')
nbcal = 10
forcecal = np.zeros(nbcal,dtype=float)
enccal = np.zeros(nbcal,dtype=int)
encMcal = np.zeros(nbcal,dtype=int)

ii=0
while ii<nbcal:
    serialPort.write(str.encode('C'))
    time.sleep(0.05)
    # Wait until there is data waiting in the serial buffer
    if(serialPort.in_waiting > 0):
        #bytesToRead = serialPort.in_waiting
        # Read data out of the buffer until a carraige return / new line is found
        serialString_f = serialPort.readline().decode('Ascii')
        serialString_enc = serialPort.readline()
        serialString_encM = serialPort.readline()
        #print(serialString)
        # Print the contents of the serial data
        forcecal[ii] = float(serialString_f)
        print(forcecal[ii])
        enccal[ii] = float(serialString_enc)
        print(enccal[ii])
        encMcal[ii] = float(serialString_encM)
        print(encMcal[ii])
        ii+=1
        #time.sleep(0.01)

baseline = statistics.mean(forcecal)
alpha0 = statistics.mean(encMcal)
z0 = statistics.mean(enccal)
print('Calibration done, baseline=',baseline, 'z0=', z0, 'alpha0=', alpha0)
input = str(baseline)
serialPort.flush()
serialPort.write(input[0:5].encode('Ascii'))
input2 = str(alpha0)
serialPort.flush()
serialPort.write(input2[0:5].encode('Ascii'))

CALIBRATION_FACTOR = 4.981002687424711 #11-03-2023
RES = 2*3.1415/5120

def recv_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])

context = zmq.Context()
#global img_socket
#img_socket = context.socket(zmq.REQ)
#img_socket.connect("tcp://172.16.0.2:5555")
flags = 0

#cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)

global run
global image_times, data_times, forcesvec, posvec, anglevec, targetvec, pwmvec, DIR_trial, video

class ZmqNode:
    def __init__(self, name, url, decoder, ntype="data"):
        self.name = name
        self.url = url
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.url)
        self.decoder = decoder
        self.ntype = ntype

    def get_data(self):
        data = recv_array(self.socket)
        return data

nodes = [
    ZmqNode("camera","tcp://172.16.0.2:5555",recv_array)
]


class Application():
    def __init__(self):
        self._force_ = 0.0
        self._position_ = 0.0
        self._angle_ = 0.0
        self._target_ = 0.0
        self._pwm_ = 0.0
        self._image_ = None
        global run
        run = True

    def start_stream(self):
        global image_times, DIR_trial, packet_q
        grab_curr_im = 0
        while run:
            
            start_time = time.time()
            packet = dict()
            for node in nodes:   
                if node.ntype == "data":
                    node.socket.send_string("sup")
                    packet[node.name] = node.get_data()
                    packet_q.put(packet)
            

                    request = time.time()
            
                    fps_images.update()
                    image_times.append(request - start_time) # current image time with respect to start in seconds
            
                grab_curr_im += 1
                if grab_curr_im % 60 == 0:
                    print(f"[INFO][CAMERA THREAD] Realtime speed {round(fps_images.fps(), 2)}")
           
    def store_data(self):
        global packet_q
        #num_images = len(packet_q)
        f = h5py.File(f"{DIR_trial}" f"many_images.h5", "w")
        while run or not packet_q.empty():
            try:
                packet = packet_q.get(timeout=1/30.)
                packet_q.task_done()

                for pk in packet.keys():
                    data = packet[pk]
                    if pk not in f.keys():
                        f.create_dataset(pk, data=data[None,...], maxshape=(None,) + data.shape)
                        #dataset = f.create_dataset("images", np.shape(data), h5py.h5t.STD_U8BE, data=data)              
                    else:
                        f[pk].resize(f[pk].shape[0] + 1, axis=0)
                        f[pk][-1] = data

            except queue.Empty:
                pass
        f.close()

    def force_acq(self):
        global image_times, data_times, forcesvec, posvec, anglevec, targetvec, pwmvec
        grab_curr = 0

        while run:
            serialPort.write(str.encode('D'))
            time.sleep(1/framerate1)
            
            # Wait until there is data waiting in the serial buffer
            if(serialPort.in_waiting > 0):

                # Read data out of the buffer until a carraige return / new line is found
                serialString_f = serialPort.readline().decode('Ascii')
                serialString_enc = serialPort.readline().decode('Ascii')
                serialString_encM = serialPort.readline().decode('Ascii')
                serialString_friction = serialPort.readline().decode('Ascii')
                serialString_target = serialPort.readline().decode('Ascii')
                serialString_U = serialPort.readline().decode('Ascii')
                
                self._force_ = (float(serialString_f)-baseline)/CALIBRATION_FACTOR
                if (self._force_<0): self._force_ = 0.0    
                self._position_ = int(serialString_enc)-z0
                self._angle_ = (float(serialString_encM)-alpha0)*RES
                self._target_ = float(serialString_target)
                self._pwm_ = float(serialString_U)

                grab_curr += 1
                if grab_curr % 10 == 0:
                    print(f"[INFO][FORCE THREAD] Data acquisition")
                    print('force:',round(self._force_,3),'position:',self._position_,'angle:',self._angle_,'friction',int(float(serialString_friction)),'voltage:',self._pwm_)


                serialPort.flush()

                
                fps_data.update()
                data_times.append(datetime.now() - start_time) # current data time with respect to start in seconds
                forcesvec.append(self._force_)
                posvec.append(self._position_)
                anglevec.append(self._angle_)
                targetvec.append(self._target_)
                pwmvec.append(self._pwm_)
            else:
                continue
        
        serialPort.write(str.encode('E'))
        time.sleep(1)

# Define the experimental plan
repeat = 5
friction_cond = np.repeat(range(4,5),repeat)
friction_random = friction_cond
random.shuffle(friction_random)
Plan = np.zeros((len(friction_random),2))
most_slippery = np.zeros(len(friction_random),dtype=int)
for ii in range(len(Plan)):
    most_slippery[ii] = int(random.randint(0,1))
    Plan[ii,most_slippery[ii]] = friction_random[ii]
answer = np.zeros(len(Plan),dtype=int) 

#Plan = np.array([[0, 1],[2, 0],[2, 0],[1, 0],[1, 2],[2, 1]])
#most_slippery = np.array([2,1,1,1,2,1])

try:
    qapp = Application()
    
    ## Initial position
    serialPort.write(str.encode('E'))
    time.sleep(1)
    print('Init done')
    input = str(0)
    serialPort.flush()
    serialPort.write(str.encode('F'))
    serialPort.flush()
    serialPort.write(input.encode('Ascii'))
    time.sleep(0.1)
    friction = serialPort.readline().decode('Ascii')

    for ii in range(len(Plan)):
        print("ii ",str(ii))
        for jj in range(2):

            ## Friction condition
            input = str(Plan[ii,jj])
            print('sending ',input,' to arduino')
            serialPort.flush()
            serialPort.write(str.encode('F'))
            serialPort.flush()
            serialPort.write(input.encode('Ascii'))
            time.sleep(0.1)
            #if(serialPort.in_waiting > 0):
            friction = serialPort.readline().decode('Ascii')
            print('friction received:',(friction))
            
            serialPort.flush()
            

            image_times = []
            data_times = []
            forcesvec = []
            posvec = []
            anglevec = []
            targetvec = []
            pwmvec = []

            if save==True:
                date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                DIR_trial = f"{home}/Data/image_test/{part}/{date_time}~{ii+1}-{jj+1}/"
                os.makedirs(DIR_trial, exist_ok=True) # Succeeds if folder already exists
                print(f"[INFO][SAVING] Made working directory: {DIR_trial}")
            
            packet_q = queue.Queue()

            thread_force_sensor = threading.Thread(target=qapp.force_acq,daemon=True)

            thread_camera = threading.Thread(target=qapp.start_stream,daemon=True)
            thread_store = threading.Thread(target=qapp.store_data, daemon=True)

            serialPort.flush()


            run = True
            fps_images = FPS().start()
            fps_data = FPS().start()
            start_time = datetime.now()
            
            print("Starting data acquisition")
            #Sending beep
            plays.play()

            thread_camera.start()
            thread_force_sensor.start()
            if save==True:
                thread_store.start()
            
            time.sleep(8)
            run = False
            plays.play()
            time.sleep(.3)
            plays.play()

            fps_images.stop()
            fps_data.stop()

            #packet_q.join()            

            # Plotting --> work on that
            #fig = plt.figure()
            #plt.plot(forcesvec)
            #plt.show()
            #plt.close()

            print("[INFO][CAMERA THREAD] elasped time: {:.2f}".format(fps_images.elapsed()))
            print("[INFO][CAMERA THREAD] approx. FPS: {:.2f}".format(fps_images.fps()))


            #Saving
            if save==True:
                filename_visiondata = join(DIR_trial, "visiondata.csv")
                with open(filename_visiondata, "w") as fi:
                    writer = csv.writer(fi)
                    writer.writerow(["image_times"])
                    writer.writerows(map(lambda x: [x], image_times))
                fi.close()

                filename_forceposition = join(DIR_trial, "force_position.csv")
                with open(filename_forceposition, "w") as fd:
                    writer = csv.writer(fd)
                    writer.writerow(["time","force","position","target","angle","pwm"])
                    for i in range(len(data_times)): 
                        writer.writerow([data_times[i], forcesvec[i], posvec[i], targetvec[i], anglevec[i], pwmvec[i]])
                fd.close()
                
            else:
                continue
        

        # Psychophysics
        correct = False
        print(f"Which surface felt the most slippery?...")
        plays_question.play()
        time.sleep(3)
        while correct == False:
            key = getch.getch()
            print(f"You pressed {key}")
            if key == 'o':
                answer[ii] = 1
                correct = True
            elif key == 't':
                answer[ii] = 2
                correct = True
            else:
                print('[WARNING] The input is incorrect, please enter again...')

        time.sleep(1)

    if save==True:
        filename_answer = join(DIR, "answer.csv")
        with open(filename_answer, "w") as fa:
            writer = csv.writer(fa)
            writer.writerow(["friction","correct","answer"])
            for i in range(len(Plan)): 
                writer.writerow([Plan[i,0]+Plan[i,1],most_slippery[i],answer[i]])
        fa.close()
    
    print("The experiment is finished")
    plays_end.play()
    time.sleep(4)

except KeyboardInterrupt:
    print('interrupted! closing')
    serialPort.write(str.encode('E'))
    time.sleep(1)
    # close all connections and windows
    serialPort.flush()
    serialPort.close()


    



