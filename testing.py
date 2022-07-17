import holistic_detector as hd
import cv2
from videos_setup import actions
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from preprocessing_and_training_NN import videos_test, labels_test
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)
detector = hd.Detector()

sequence = []
model = load_model('pose_gesture_model')

while cap.isOpened():
    # reading the image from video capture
    _, img = cap.read()
    img = detector.init_landmarks(img)

    key_points = detector.get_landmarks()
    # prediction logic, waits till 30 frames worth of key points are stacked
    sequence.append(key_points)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])

    cv2.imshow("Vision", img)

    # breaks out of the loop if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

prediction = model.predict(videos_test)
prediction_true = np.argmax(labels_test, axis=1).tolist()
prediction = np.argmax(prediction, axis=1).tolist()
print(multilabel_confusion_matrix(prediction_true, prediction))
print(accuracy_score(prediction_true, prediction))

cap.release()
cv2.destroyAllWindows()
