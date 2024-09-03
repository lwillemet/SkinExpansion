# SkinExpansion
Author: Laurence Willemet, MIT.

## Description
This folder includes the codes used to acquire the dataset of human experiment with the skin expansion setup.
In this work, we built a tactile device that radially deforms both the index and thumb using soft deformable membranes. The slipperiness sensation is achieved by modulating the rate of expansion, which can be controlled according to the squeezing force applied on the object. The results suggest that this deformation pattern generates a sensation of slipperiness. These results can significantly improve the rendering of the haptic modality in tele-robotics, where operators control robot manipulators over a distance.

## Pre-requists:
Requirred python libraries:
- communication Arduino: serial,
- image acquisition: zmq,
- image saving: h5py,
- multithreading: threading,
- user inputs: getch

## Before running the experiment:
- Plug in the power supply and the LEDs to the RPi.
- Connect the 3 USB cables to the computer.
- Communication with then Arduino:
  copy/paste this line in a terminal: sudo chmod a+rw /dev/ttyACM0
- Communication with the Raspberry Pi:
    ssh pi@172.16.0.2
    password: BaxterUR555
  start camera streamer: python3 /home/pi/zmq_node.py
  wait until it is Running... in the terminal

## Codes
1) camera_test.py was used to check the image visualization in real time from the raspberry pi.
2) familiarization.py was run first so the participant can familiarised theirself with the setup. It consists of 5 trials, so a total of 10 lifts.
3) dataAcquisition.py is running the real experiment of 50 trials in a row (So 100 lifts).

After 2 lifts, the code is waiting for the participant response. It is the only opportunity for the participant to take a break (wait before answering).
