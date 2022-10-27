import numpy as np
import cv2
from yolo_nn import Yolo


class KalmanFilter:

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time
        self.dt = dt
        # Define the  control input variables
        self.u = np.matrix([[u_x], [u_y]])
        # Initial State
        self.x = np.matrix([[0], [0], [0], [0]])
        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])
        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Update time state
        self.x = self.A.dot(self.x) + self.B.dot(self.u)
        # Calculate error covariance
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
        return self.x[0], self.x[1]

    def update(self, z):
        z = np.array(z).reshape((2, 1))
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        # Calculate the Kalman Gain
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K*(z - self.H.dot(self.x))

        self.P = self.P - K.dot(self.H).dot(self.P)
        return self.x[0], self.x[1]


def main():
    # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    KF = KalmanFilter(0.15, 0, 0, 1, 0.1, 0.1)
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture("test.mp4")
    yolo = Yolo()

    while True:
        # Read frame
        ret, frame = VideoCap.read()

        yolo.track(frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            init_box = cv2.selectROI("Select object for tracking", frame, fromCenter=False, showCrosshair=False)
            yolo.init_tracking(init_box)
            cv2.destroyWindow("Select object for tracking")

        if yolo.tracked_to is not None:
            center = yolo.tracked_to.centroid

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)
            # Update
            (x1, y1) = KF.update(center)
            cv2.circle(frame, (int(x1), int(y1)), 1, (0, 0, 255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
