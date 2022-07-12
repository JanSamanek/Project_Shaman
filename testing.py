import holistic_detector as hd
import cv2
import numpy as np
import building_and_training_NN as NN
from folder_setup import actions

cap = cv2.VideoCapture(0)    # TODO remove inbuilt camera view
detector = hd.Detector()
sequence = []

while cap.isOpened():
    # reading the image from video capture
    _, img = cap.read()
    img = detector.init_landmarks(img)

    key_points = detector.get_landmarks()
    # prediction logic, waits till 30 frames worth of key points are stacked
    sequence.append(key_points)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = NN.model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])

    cv2.imshow("Vision", img)

    # breaks out of the loop if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
