import numpy as np
import cv2 as cv


class KalmanFilter:

    def __init__(self):
        # init kf with 4 dynamic params (x, y, dx, dy) and 2 measured params (x, y)
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementNoiseCov = np.array([[5, 0], [0, 5]], np.float32)

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


def main():
    # Kalman Filter
    kf = KalmanFilter()
    img = cv.imread("Bb_blue.jpg")
    ball_positions = [(50, 100), (100, 100), (150, 100), (200, 100), (250, 100), (300, 100), (350, 100), (400, 100), (450, 100)]
    for pt in ball_positions:
        cv.circle(img, pt, 15, (0, 20, 220), -1)
        predicted = kf.predict(pt[0], pt[1])
        cv.circle(img, predicted, 15, (20, 220, 0), 4)

    cv.imshow("Img", img)
    cv.waitKey(1)
    input()


if __name__ == '__main__':
    main()
