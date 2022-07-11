import os.path
import numpy as np
import cv2
import pose_detector as pd
from folder_setup import actions, video_num, video_len, DATA_PATH

cap = cv2.VideoCapture(0)
detector = pd.Detector()

for action in actions:
    # loop through videos
    for video in range(video_num):
        # loop through video_length
        for frame_num in range(video_len):

            # reading the image from video capture
            _, img = cap.read()

            img = detector.init_landmarks(img)

            # breaks out of the loop if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # collection logic
            if frame_num == 0:
                cv2.putText(img, 'Starting collecting', (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(img, f'Collecting frames for {action}. Video number {video}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("", img)
                cv2.waitKey(2000)
            else:
                cv2.putText(img, f'Collecting frames for {action}. Video number {video}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("", img)

            lm_gesture = detector.get_landmarks(img)
            path = os.path.join(DATA_PATH, action, str(video), str(frame_num))
            np.save(path, lm_gesture)

cap.release()
cv2.destroyAllWindows()
