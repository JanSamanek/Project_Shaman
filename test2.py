import cv2
import numpy as np
from Utilities.display import display_fps

dt = 4
A = np.matrix([[1, 0, dt, 0],
               [0, 1, 0, dt],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

dt=5
print(A)