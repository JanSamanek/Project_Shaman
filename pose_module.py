import cv2
import mediapipe as mp
import time
import math


class PoseDetector:

    def __init__(self, **kwargs):

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**kwargs)

    def find_pose(self, img, draw=True):

        # detecting pose and drawing landmarks, connections
        # cv2 reads the image in BGR but mp needs RGB as input
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_RGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):

        self.landmark_list = []

        if self.results.pose_landmarks:
            # getting list of landmarks with corresponding coordinates
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = img.shape
                # getting pixel value position of the landmarks
                position_pix_x, position_pix_y = int(landmark.x*width), int(landmark.y*height)
                self.landmark_list.append((landmark_id, position_pix_x, position_pix_y))
                if draw:
                    cv2.circle(img, (position_pix_x, position_pix_y), 3, (255, 0, 0), cv2.FILLED)

        return self.landmark_list

    def find_angle(self, img, point1, point2, point3, draw=True):

        # retrieve x, y position for each point
        try:
            position_pix_x1, position_pix_y1 = self.landmark_list[point1][1:]       # TODO indexing out of range problem ? solve
            position_pix_x2, position_pix_y2 = self.landmark_list[point2][1:]
            position_pix_x3, position_pix_y3 = self.landmark_list[point3][1:]
        except IndexError:
            print("No parameters to calculate angle")
        else:
            # calculate angle
            angle = math.degrees(math.atan2(position_pix_y3 - position_pix_y2, position_pix_x3 - position_pix_x2)
                                 - math.atan2(position_pix_y1 - position_pix_y2, position_pix_x1 - position_pix_x2))

            # draw
            if draw:
                cv2.line(img, (position_pix_x1, position_pix_y1), (position_pix_x2, position_pix_y2), (0, 0, 255), 3)
                cv2.line(img, (position_pix_x2, position_pix_y2), (position_pix_x3, position_pix_y3), (0, 0, 255), 3)
                cv2.circle(img, (position_pix_x1, position_pix_y1), 3, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (position_pix_x1, position_pix_y1), 6, (0, 0, 255))
                cv2.circle(img, (position_pix_x2, position_pix_y2), 3, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (position_pix_x2, position_pix_y2), 6, (0, 0, 255))
                cv2.circle(img, (position_pix_x3, position_pix_y3), 3, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (position_pix_x3, position_pix_y3), 6, (0, 0, 255))

            return angle


def main():
    previous_time = 0

    vision_cap = cv2.VideoCapture(0)    # TODO remove inbuilt camera view
    detector = PoseDetector()

    while True:
        # reading the image from video capture
        _, img = vision_cap.read()
        img = detector.find_pose(img)

        lm_list = detector.find_position(img)
        detector.find_angle(img, 12, 14, 16)
        # print(lm_list[12])

        # measuring and displaying fps
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 3)

        cv2.imshow("Vision", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()