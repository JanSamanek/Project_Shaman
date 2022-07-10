import mediapipe as mp
import cv2

mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()  # TODO could use with to also destroy the object when not used

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_RGB)

    print(results.pose_landmarks)

    cv2.imshow("Robot's eyes", img)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
