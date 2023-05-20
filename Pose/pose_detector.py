import cv2
import mediapipe as mp
import math
from Utilities.display import Utility_helper

class PoseDetector:
    LEFT_HAND = 19
    LEFT_SHOULDER = 11
    LEFT_HIP = 23
    LEFT_ELBOW = 13
    RIGHT_HAND = 20
    RIGHT_SHOULDER = 12
    RIGHT_HIP = 24
    RIGHT_ELBOW = 14
    NOSE = 0
    LM_X = 1
    LM_Y = 2

    def __init__(self, **kwargs):
        print("[INF] Initiliazing pose detector ...")
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**kwargs)
        self.pose_landmarks = []
        self.crossed_hands_counter = 0

    def get_landmarks(self, img, box=None, draw=True):
        lm_list = []

        if box is not None:
            img = PoseDetector._crop_im(img, *box)

        img.flags.writeable = False     # to enhance performance
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # cv2 reads the image in BGR but mp needs RGB as input
        self.results = self.pose.process(img_RGB)
        img.flags.writeable = True

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_draw.DrawingSpec(color=(61, 254, 96), thickness=3, circle_radius=2),
                                            self.mp_draw.DrawingSpec(color=(253, 63, 28), thickness=3, circle_radius=2))
                
        if self.results.pose_landmarks:
            for ID, lm in enumerate(self.results.pose_landmarks.landmark):
                lm_list.append([ID, lm.x, lm.y, lm.z, lm.visibility])
            self.pose_landmarks = lm_list
        return img

    def _detect_right_hand_above_nose(self):
        if self.pose_landmarks[PoseDetector.RIGHT_HAND][PoseDetector.LM_Y] < self.pose_landmarks[PoseDetector.NOSE][PoseDetector.LM_Y]:
            return True
        else:
            return False

    def _detect_left_hand_above_nose(self):
        if self.pose_landmarks[PoseDetector.LEFT_HAND][PoseDetector.LM_Y] < self.pose_landmarks[PoseDetector.NOSE][PoseDetector.LM_Y]:
            return True
        else:
            return False

    def _detect_right_hand_elavated(self):
        if 100 >= self._detect_angle(PoseDetector.RIGHT_HIP, PoseDetector.RIGHT_SHOULDER, PoseDetector.RIGHT_ELBOW) >= 60\
            and 250 >= self._detect_angle(PoseDetector.RIGHT_SHOULDER, PoseDetector.RIGHT_ELBOW, PoseDetector.RIGHT_HAND) >= 180:
            return True
        else:
            return False
    
    def _detect_left_hand_elavated(self):
        if 310 >= self._detect_angle(PoseDetector.LEFT_HIP, PoseDetector.LEFT_SHOULDER, PoseDetector.LEFT_ELBOW) >= 270\
            and 180 >= self._detect_angle(PoseDetector.LEFT_SHOULDER, PoseDetector.LEFT_ELBOW, PoseDetector.LEFT_HAND) >= 110:
            return True
        else:
            return False
        
    def _detect_crossed_hands(self):
        if self.pose_landmarks[PoseDetector.RIGHT_HAND][PoseDetector.LM_X] > self.pose_landmarks[PoseDetector.LEFT_HAND][PoseDetector.LM_X]:
            self.crossed_hands_counter += 1
            if self.crossed_hands_counter > 3:
                self.crossed_hands_counter = 0
                return True
            else:
                return False
        else:
            self.crossed_hands_counter = 0
            return False


    def _detect_angle(self, point1, point2, point3):

        # retrieve x, y position for each point
        position_pix_x1, position_pix_y1 = self.pose_landmarks[point1][1:3]
        position_pix_x2, position_pix_y2 = self.pose_landmarks[point2][1:3]
        position_pix_x3, position_pix_y3 = self.pose_landmarks[point3][1:3]

        # calculate angle
        angle = math.degrees(math.atan2(position_pix_y3 - position_pix_y2, position_pix_x3 - position_pix_x2)
                                - math.atan2(position_pix_y1 - position_pix_y2, position_pix_x1 - position_pix_x2))
        angle = angle + 360 if angle < 0 else angle

        return angle
  
    def get_gestures(self, img, box=None):
        self.get_landmarks(img, box, draw=False)
        gestures = {}
        # can i use get instead of try?
        try:
            left_hand_up =  self._detect_left_hand_above_nose()
            right_hand_up = self._detect_right_hand_above_nose()
            left_hand_elevated = self._detect_left_hand_elavated()
            right_hand_elevated = self._detect_right_hand_elavated()
            crossed_hands = self._detect_crossed_hands()
        except IndexError as e:
            print("[ERROR] Gesture list: ", e)
        else:
            if left_hand_up and right_hand_up:
                if crossed_hands:
                    gestures["crossed"] = True
                else:
                    gestures["both_up"] = True
            elif left_hand_up:
                gestures["left_up"] = True
            elif right_hand_up:
                gestures["right_up"] = True
            elif right_hand_elevated:
                gestures["right_elevated"] = True
            elif left_hand_elevated:
                gestures["left_elevated"] = True
        
        return gestures

    @staticmethod
    def _crop_im(img, start_x, start_y, end_x, end_y):
        # enlarge the crop
        new_start_y = int(start_y * 0.85)
        new_end_y = int(end_y * 1.1)
        new_start_x = int(start_x * 0.92)
        new_end_x = int(end_x * 1.08)

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
        
        previous_time = Utility_helper.display_fps(img , previous_time)
        gestures = detector.get_gestures(img)         
        
        if gestures.get("both_up", False):
            print("both up")
        elif gestures.get("left_up", False):
            print("left up")
        elif gestures.get("right_up", False):
            print("right_up")
        elif gestures.get("right_elevated", False):
            print("right_elevated")
        elif gestures.get("left_elevated", False):
            print("left_elevated")
        elif gestures.get("crossed", False):
            print("crossed")
        else:
            print("Nothing")

        cv2.imshow("Vision", img)

        # breaks out of the loop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def pose_img_prediction(img_path, save_path="pose_showcase.jpg"):
    img = cv2.imread(img_path)
    pose_detector = PoseDetector()
    img = pose_detector.get_landmarks(img)
    cv2.imwrite(save_path, img)
    return img


if __name__ == '__main__':
    main()