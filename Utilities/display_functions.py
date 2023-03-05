import time
import cv2 as cv


def display_fps(img, previous_time):
    # measuring and displaying fps
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    string = "FPS: " + str(int(fps))
    cv.putText(img, string, (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    return previous_time


def display_motor_speed(img, motor_speed_one, motor_speed_two):
    text_one = "Motor ONE: " + str(motor_speed_one)
    text_two = "Motor TWO: " + str(motor_speed_two)
    cv.putText(img, text_one, (50, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv.putText(img, text_two, (50, 110), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
