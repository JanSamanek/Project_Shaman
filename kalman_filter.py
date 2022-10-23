import numpy as np
import cv2 as cv


class KalmanFilter:

    def __init__(self, init_state):
        # init kf with 4 dynamic params (x, y, dx, dy) and 2 measured params (x, y)
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # self.kf.measurementNoiseCov = np.array([[5, 0], [0, 5]], np.float32)
        self.init_state = np.array([[np.float32(init_state[0])], [np.float32(init_state[1])]])
        self.first_measure = True

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        if self.first_measure:
            self.kf.correct(np.subtract(measured, self.init_state))
            predicted = self.kf.predict()
            x, y = int(predicted[0] + self.init_state[0]), int(predicted[1] + self.init_state[0])
            self.first_measure = False
            return x, y

        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])

        return x, y


def main():
    # Kalman Filter
    kf = KalmanFilter((100, 150))
    img = cv.imread("blue_backround.png")
    ball_positions = [(100, 150), (150, 140), (200, 130), (250, 120), (300, 110), (350, 100), (400, 100), (450, 100), (500, 100)]
    predicted = None

    for i, pt in enumerate(ball_positions):
        cv.circle(img, pt, 15, (0, 20, 220), -1)
        predicted = kf.predict(pt[0], pt[1])
        if i == 1:
            cv.circle(img, predicted, 15, (255, 220, 0), 4)
        else:
            cv.circle(img, predicted, 15, (20, 220, 0), 4)

    for i in range(10):
        predicted = kf.predict(predicted[0], predicted[1])
        cv.circle(img, predicted, 15, (20, 220, 0), 4)

    cv.imshow("Img", img)
    cv.waitKey(1000000)



if __name__ == '__main__':
    main()
