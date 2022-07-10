import cv2
import mediapipe as mp
import time
import math


class LandmarkDetector:

    def __init__(self, **kwargs):

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(**kwargs)

        self.lm_dict = {}

    def detect_landmarks(self, img, draw=True):

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

    def get_landmarks(self, img, body_part):

        r_hand_lm_list = []
        l_hand_lm_list = []
        pose_lm_list = []

        height, width, _ = img.shape

        if self.results.pose_landmarks:
            # getting list of landmarks with corresponding coordinates
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                # getting pixel value position of the landmarks
                position_pix_x, position_pix_y = int(landmark.x*width), int(landmark.y*height)
                pose_lm_list.append((landmark_id, position_pix_x, position_pix_y))
        self.lm_dict["pose"] = pose_lm_list

        if self.results.right_hand_landmarks:
            # getting list of landmarks with corresponding coordinates
            for landmark_id, landmark in enumerate(self.results.right_hand_landmarks.landmark):
                # getting pixel value position of the landmarks
                position_pix_x, position_pix_y = int(landmark.x*width), int(landmark.y*height)
                r_hand_lm_list.append((landmark_id, position_pix_x, position_pix_y))
        self.lm_dict["right_hand"] = r_hand_lm_list

        if self.results.left_hand_landmarks:
            # getting list of landmarks with corresponding coordinates
            for landmark_id, landmark in enumerate(self.results.left_hand_landmarks.landmark):
                # getting pixel value position of the landmarks
                position_pix_x, position_pix_y = int(landmark.x*width), int(landmark.y*height)
                l_hand_lm_list.append((landmark_id, position_pix_x, position_pix_y))
        self.lm_dict["left_hand"] = l_hand_lm_list

        return self.lm_dict[body_part]


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
    detector = LandmarkDetector()
    previous_time = 0

    while True:
        # reading the image from video capture
        _, img = vision_cap.read()
        img = detector.detect_landmarks(img, False)

        print(detector.get_landmarks(img, 'right_hand'))  # TODO Index out of range problem if body part isn't in the frame

        previous_time = display_fps(img, previous_time)

        cv2.imshow("Vision", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
