import cv2
import mediapipe as mp
import time
import numpy as np


class HolisticDetector:

    POSE_LM_NUM = 33
    HAND_LM_NUM = 21
    FACE_LM_NUM = 468

    def __init__(self, **kwargs):

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(**kwargs)

    def init_landmarks(self, img, draw=True):

        # to enhance performance
        img.flags.writeable = False
        # detecting pose and drawing landmarks, connections
        # cv2 reads the image in BGR but mp needs RGB as input
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(img_RGB)
        img.flags.writeable = True

        if draw:
            if self.results.right_hand_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if self.results.left_hand_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if self.results.pose_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            if self.results.face_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                            self.mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                            self.mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        return img

    def get_landmarks(self):

        r_hand_lm_list = []
        l_hand_lm_list = []
        pose_lm_list = []
        face_lm_list = []

        if self.results.pose_landmarks:
            for lm in self.results.pose_landmarks.landmark:
                pose_lm_list.append([lm.x, lm.y, lm.z, lm.visibility])
            pose_landmarks = np.array(pose_lm_list).flatten()
        else:
            pose_landmarks = np.zeros(4 * HolisticDetector.POSE_LM_NUM)

        if self.results.right_hand_landmarks:
            for lm in self.results.right_hand_landmarks.landmark:
                r_hand_lm_list.append([lm.x, lm.y, lm.z])
            r_hand_landmarks = np.array(r_hand_lm_list).flatten()
        else:
            r_hand_landmarks = np.zeros(3 * HolisticDetector.HAND_LM_NUM)

        if self.results.left_hand_landmarks:
            for lm in self.results.left_hand_landmarks.landmark:
                l_hand_lm_list.append([lm.x, lm.y, lm.z])
            l_hand_landmarks = np.array(l_hand_lm_list).flatten()
        else:
            l_hand_landmarks = np.zeros(3 * HolisticDetector.HAND_LM_NUM)

        if self.results.face_landmarks:
            for lm in self.results.face_landmarks.landmark:
                face_lm_list.append([lm.x, lm.y, lm.z])
            face_landmarks = np.array(face_lm_list).flatten()
        else:
            face_landmarks = np.zeros(3 * HolisticDetector.FACE_LM_NUM)

        self.landmarks = np.concatenate([pose_landmarks, face_landmarks, l_hand_landmarks, r_hand_landmarks])

        return self.landmarks


def display_fps(img, previous_time):

    # measuring and displaying fps
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 3)

    return previous_time


def main():

    vision_cap = cv2.VideoCapture(0)    # TODO remove inbuilt camera view
    detector = HolisticDetector()
    previous_time = 0

    while vision_cap.isOpened():
        # reading the image from video capture
        _, img = vision_cap.read()
        img = detector.init_landmarks(img)

        detector.get_landmarks()

        previous_time = display_fps(img, previous_time)
        cv2.imshow("Vision", img)

        # breaks out of the loop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
