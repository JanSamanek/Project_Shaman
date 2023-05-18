# Project_Shaman
This project is my bachelor's thesis that focuses on creating a mobile robot that will be able to follow a human while avoiding obstacles and reacting to commands given through gestures.

The Detector folder contains the code that is used for detecting people in an image.

The folder Pose handles pose estimation and gesture recognition.

The TrackerBase folder contains a Kalman filtr implementation together with a TrackableObject class that is used to represent tracked persons, the center_tracker file is the heart of the tracking algorithm.

The Tracker folder contains the actual tracker that combines person detection and the tracking algorithm together.

The Controller folder contains a RobotController class which purpose is to calculate the duty cycle for the motors in order to make the robot follow a specific person, it also implements the gesture detection making use of the pose detector.

The communication.py file handles communication (sending instructions) between the robot and a distant computer.

In order to make the program work you have to install opencv with g-streamer support on the distant computer, along with mosquitto, json, mediapipe, numpy and yolov5 dependecies. On the robot, namely JetBot, you have to install g-streamer if not already installed and the jetbot library and the mpu6050 library. Then you run the communication.py file on jetbot with the argument -d jetbot -ip ip_adress_to_connect_to: ```communication.py -d jetbot -ip ip_adress_to_connect_to``` and on a distant computer you run the same script with the -d computer argument: ```communication.py -d computer```. because a mosquito server has to be created first, you have to run the communication script on the distant computer first.
