# Project_Shaman
This project is my bachelor's thesis that focuses on creating a mobile robot that will be able to follow a human while avoiding obstacles and reacting to commands given through gestures.

## Folder management
The Detector folder contains the code that is used for detecting people in an image.

The folder Pose handles pose estimation and gesture recognition.

The TrackerBase folder contains a Kalman filtr implementation together with a TrackableObject class that is used to represent tracked persons, the center_tracker file is the heart of the tracking algorithm.

The Tracker folder contains the actual tracker that combines person detection and the tracking algorithm together.

The Controller folder contains a RobotController class which purpose is to calculate the duty cycle for the motors in order to make the robot follow a specific person, it also implements the gesture detection making use of the pose detector.if you 

The communication.py file handles communication (sending instructions) between the robot and a distant computer.

## Installation instructions
In order to make the program work you have to install on the distant computer:
* opencv with g-streamer support from source
* mqtt, mediapipe, numpy:
 
```pip install paho-mqtt, mediapipe, numpy```

* yolov5 dependecies:

 ```git clone https://github.com/ultralytics/yolov5```
 
 ```cd yolov5```                          
 
 ```pip install -r requirements.txt```
                            
 You have to change the config file of mqtt to enable outside connections.
                            
You have to setup JetBot with a prebuild image -> https://jetbot.org/master/software_setup/sd_card.html.

Next you need to install

* the mpu6050 library:

```pip3 install adafruit-blinka, adafruit-circuitpython-mpu6050```

* mqtt 
 
```sudo apt-get install mosquitto mosquitto-clients``` 
 
 ## Program starting
Then you run the communication.py file on jetbot with the argument -d jetbot -ip ip_adress_to_connect_to: 

```communication.py -d jetbot -ip ip_adress_to_connect_to``` 

and on a distant computer you run the same script with the -d computer argument: 

```communication.py -d computer```.

Because a mosquito server has to be created first, you have to run the communication script on the distant computer first.

## Robot Control
After succesfully establishing the communication between the distant computer and jetbot you can initialize the robot to turn to you and listen to your command given through gestures by raising your right arm to a 90 degree position. If you want to make the robot follow you around keep you left arm above your head. If you want the robot to move straight foward keep you right arm raised above your head. If you want the robot to move straight backwards raise your left arm to a 90 degree position. By crossing your hands above your head you stop the program, that means that the connection is closed and the robot doesn't listen to your commands anymore. You have to establish a new connection  if you want the robot to follow you again.
