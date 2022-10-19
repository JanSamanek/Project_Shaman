import numpy as np


class KalmanFilter:

    def __init__(self):

        dt = np.NAN
        # two element state vector for position and velocity
        self.x = np.array([0, 0])

        # state transition matrix
        self.A = np.array([[1, dt],
                           [0, 1]])

        # state covariance matrix representing the uncertainty in x
        self.P = np.array([[5, 0],
                           [0, 5]])

        # state to measurement transition matrix
        self.H = np.array([[1, 0]])

        # H transposed
        self.HT = np.transpose(self.H)

        # input measurement variance
        self. R = 10

        # system noise covariance matrix, accounts for inaccuracy in the system model
        self.Q = np.array([[1, 0],
                           [0, 3]])

    def filter(self, pos, dt):    # TODO dt problem

        # the time of measurement will differ based on fps
        self._update_dt(dt)

        # Predict State Forward
        x_p = self.A.dot(self.x)

        # Predict Covariance Forward
        P_p = self.A.dot(self.P).dot(self.A.T) + self.Q

        # Compute Kalman Gain
        S = self.H.dot(P_p).dot(self.HT) + self.R
        K = P_p.dot(self.HT)*(1/S)

        # Estimate State
        residual = pos - self.H.dot(x_p)
        self.x = x_p + K*residual

        # Estimate Covariance
        self.P = P_p - K.dot(self.H).dot(P_p)

        return self.x[0], self.x[1], self.P

    def _update_dt(self, dt):
        self.A = self.A = np.array([[1, dt],
                                    [0, 1]])


if __name__ == '__main__':
    pass