import cv2
import mediapipe as mp
import time
import math

class PoseDetector:

    def __init__(self, **kwargs):

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**kwargs)
        self.pose_landmarks = []

    def init_landmarks(self, img, draw=True):

        # to enhance performance
        img.flags.writeable = False
        # detecting pose and drawing landmarks, connections
        # cv2 reads the image in BGR but mp needs RGB as input
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_RGB)
        img.flags.writeable = True

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def get_landmarks(self):
        lm_list = []
        if self.results.pose_landmarks:
            for ID, lm in enumerate(self.results.pose_landmarks.landmark):
                lm_list.append([ID, lm.x, lm.y])
            self.pose_landmarks = lm_list
        return self.pose_landmarks

    def detect_right_hand_above_nose(self):
        RIGHT_HAND_NUM = 16
        NOSE_NUM = 0
        LM_Y = 2

        try:
            if self.pose_landmarks[RIGHT_HAND_NUM][LM_Y] < self.pose_landmarks[NOSE_NUM][LM_Y]:
                return True
            else:
                return False
        except IndexError:
            print("missing hand or nose in video")

    def detect_left_hand_above_nose(self):
        LEFT_HAND_NUM = 15
        NOSE_NUM = 0
        LM_Y = 2

        try:
            if self.pose_landmarks[LEFT_HAND_NUM][LM_Y] < self.pose_landmarks[NOSE_NUM][LM_Y]:
                return True
            else:
                return False

        except IndexError:
            print("missing hand or nose in video")

    def detect_angle(self, img, point1, point2, point3):

        # retrieve x, y position for each point
        try:
            position_pix_x1, position_pix_y1 = self.pose_landmarks[point1][1:]
            position_pix_x2, position_pix_y2 = self.pose_landmarks[point2][1:]
            position_pix_x3, position_pix_y3 = self.pose_landmarks[point3][1:]
        except IndexError:
            print("missing parameters to calculate angle")
        else:
            # calculate angle
            angle = math.degrees(math.atan2(position_pix_y3 - position_pix_y2, position_pix_x3 - position_pix_x2)
                                 - math.atan2(position_pix_y1 - position_pix_y2, position_pix_x1 - position_pix_x2))
            if angle < 0:
                angle += 360
            return angle
  
    @staticmethod
    def crop_im(img, start_x, start_y, end_x, end_y):
        # enlarge the crop
        new_start_y = int(start_y * 0.85)
        new_end_y = int(end_y * 1.1)
        new_start_x = int(start_x * 0.98)
        new_end_x = int(end_x * 1.02)

        if new_end_y > img.shape[0]:
            new_end_y = img.shape[0]
        if new_end_x > img.shape[1]:
            new_end_x = img.shape[1]

        return img[new_start_y:new_end_y, new_start_x:new_end_x]
    
    
def main():
    previous_time = 0

    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while cap.isOpened():
        # reading the image from video capture
        _, img = cap.read()
        img = detector.init_landmarks(img)

        pose_list = detector.get_landmarks()

        cv2.imshow("Vision", img)

        # breaks out of the loop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()