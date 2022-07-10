import cv2
import mediapipe as mp
import time
import math


class Detector:

    def __init__(self, **kwargs):

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(**kwargs)

    def detect(self, img, draw=True):

        # detecting pose and drawing landmarks, connections
        # cv2 reads the image in BGR but mp needs RGB as input
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(img_RGB)
        if draw:
            if self.results.right_hand_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if self.results.left_hand_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if self.results.pose_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        return img

    # def detect_position(self, img, draw=True):
    #
    #     self.landmark_list = []
    #
    #     if self.results:
    #         # getting list of landmarks with corresponding coordinates
    #         for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
    #             height, width, _ = img.shape
    #             # getting pixel value position of the landmarks
    #             position_pix_x, position_pix_y = int(landmark.x*width), int(landmark.y*height)
    #             self.landmark_list.append((landmark_id, position_pix_x, position_pix_y))
    #             if draw:
    #                 cv2.circle(img, (position_pix_x, position_pix_y), 3, (255, 0, 0), cv2.FILLED)
    #
    #     return self.landmark_list


def main():

    vision_cap = cv2.VideoCapture(0)    # TODO remove inbuilt camera view
    detector = Detector()

    while True:
        # reading the image from video capture
        _, img = vision_cap.read()
        img = detector.detect(img)

        # print(lm_list[12])

        cv2.imshow("Vision", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
