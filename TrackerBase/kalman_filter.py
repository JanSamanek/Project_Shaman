import numpy as np


class KalmanFilter:

    def __init__(self, init_pos, dt=0.08, u_x=0, u_y=0, u_z=0, std_acc=1, x_std_meas=0.4, y_std_meas=0.4, FoV=160, img_width=1280):
        """
        :param init_pos: initial position (x,y)
        :param camera_rotation: the rotation of camera around the z axis
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param u_z: angular speed of the camera around the z-axis [rad/s]
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        :param FoV: field of view of the camera used [°]
        """
        #define field of view
        self.FoV = (FoV/360)*2*np.pi

        # Define sampling time
        self.dt = dt

        # Define the control input variables
        self.u = np.matrix([[u_x], [u_y], [u_z]])

        # Initial State
        self.x = np.matrix([[init_pos[0]], [init_pos[1]], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0, self.dt*img_width/self.FoV],
                            [0, (self.dt**2)/2, 0], 
                            [self.dt, 0, 0],        ### u_z*R ?????
                            [0, self.dt, 0]])
        
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

    def predict(self, camera_rotation=0):
        self.u = np.matrix([[0], [0], [camera_rotation]]) 
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
        