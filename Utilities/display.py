import time
import cv2 as cv
import json
import numpy as np


with open("settings.json") as json_file:
    data = json.load(json_file)
    FoV = data['FoV']
    dt = data['dt']
    img_width = data['img_width']


class Utility_helper():

    @staticmethod
    def display_fps(img, previous_time):
        # measuring and displaying fps
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        string = "FPS: " + str(int(fps))
        cv.putText(img, string, (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        return previous_time

    @staticmethod
    def display_motor_speed(img, motor_speed_one, motor_speed_two):
        text_one = "Motor ONE: " + str(motor_speed_one)
        text_two = "Motor TWO: " + str(motor_speed_two)
        cv.putText(img, text_one, (50, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        cv.putText(img, text_two, (50, 110), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    @staticmethod
    def display_camera_rotation(img, camera_rotation):
        text = 'camera shift: ' + str(int(dt*img_width*camera_rotation/((FoV*np.pi*2)/360))) + ' pixels'
        cv.putText(img, text, (50, 140), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
