import numpy as np


class KalmanFilter:

    def __init__(self, init_pos, dt=0.15, u_x=0, u_y=0, std_acc=1, x_std_meas=0.4, y_std_meas=0.4):
        """
        :param init_pos: initial position (x,y)
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
        self.x = np.matrix([[init_pos[0]], [init_pos[1]], [0], [0]])
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
        return self.x.item(0), self.x.item(1)

    def update(self, z):
        z = np.array(z).reshape((2, 1))
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        # Calculate the Kalman Gain
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K*(z - self.H.dot(self.x))

        self.P = self.P - K.dot(self.H).dot(self.P)
        return self.x.item(0), self.x.item(1)


def main():
    import cv2
    from yolo_nn import Yolo
    import matplotlib
    import matplotlib.pyplot as plt

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture("test.mp4")
    yolo = Yolo()
    measured_pos = []
    predicted_pos = []
    filtered_pos = []

    while VideoCap.isOpened():
        # Read frame
        ret, frame = VideoCap.read()

        yolo.track(frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            init_box = cv2.selectROI("Select object for tracking", frame, fromCenter=False, showCrosshair=False)
            yolo.init_tracking(init_box)
            cv2.destroyWindow("Select object for tracking")
            # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
            KF = KalmanFilter(_calculate_center(*init_box))

        if yolo.tracked_to is not None:
            center = yolo.tracked_to.centroid
            measured_pos.append(center)

            # Predict
            (x, y) = KF.predict()
            predicted_pos.append((x, y))
            # Draw a rectangle as the predicted object position
            cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)

            # Update
            (x1, y1) = KF.update(center)
            filtered_pos.append((x1, y1))
            cv2.circle(frame, (int(x1), int(y1)), 1, (0, 0, 255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

    matplotlib.use('TkAgg')
    plt.title('Kalman filter', fontsize=20)

    plt.plot(np.array(measured_pos)[:, 0], np.array(measured_pos)[:, 1], label='measured position')
    plt.plot(np.array(predicted_pos)[:, 0], np.array(predicted_pos)[:, 1], label='predicted position')
    plt.plot(np.array(filtered_pos)[:, 0], np.array(filtered_pos)[:, 1], label='filtered position')

    plt.legend()
    plt.show()
    input()


def _calculate_center(start_x, start_y, width, height):
    center_x = int((start_x + (start_x + width)) / 2.0)
    center_y = int((start_y + (start_y + height)) / 2.0)
    return center_x, center_y


if __name__ == '__main__':
    main()
