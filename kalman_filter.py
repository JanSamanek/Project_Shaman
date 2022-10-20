import random
import numpy as np
import cv2 as cv


class KalmanFilter:

    def __init__(self):
        # init kf with 4 dynamic params (x, y, dx, dy) and 2 measured params (x, y)
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


def main():
    pass


if __name__ == '__main__':
    main()
