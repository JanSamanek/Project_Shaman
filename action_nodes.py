from pose_detector import PoseDetector
from flow_nodes import NodeStates, Node, Fallback
import cv2 as cv


class RightHandAboveCheck(Node):

    def __init__(self, *children_nodes, cap):
        super().__init__(*children_nodes)
        self.cap = cap
        self.detector = PoseDetector()

    def evaluate(self):

        # reading the image from video capture
        _, img = self.cap.read()

        img = self.detector.init_landmarks(img)

        self.detector.get_landmarks(img)

        cv.waitKey(1)

        if self.detector.detect_right_hand_above_nose():
            print("Right Hand: Success")
            return NodeStates.SUCCESS
        else:
            print("Right Hand: Failure")
            return NodeStates.FAILURE


class LeftHandAboveCheck(Node):

    def __init__(self, *children_nodes, cap):
        super().__init__(*children_nodes)
        self.cap = cap
        self.detector = PoseDetector()

    def evaluate(self):

        # reading the image from video capture
        _, img = self.cap.read()

        img = self.detector.init_landmarks(img)

        self.detector.get_landmarks(img)

        cv.waitKey(1)

        if self.detector.detect_left_hand_above_nose():
            print("Left Hand: Success")
            return NodeStates.SUCCESS
        else:
            print("Left Hand: Failure")
            return NodeStates.FAILURE


# # error when inside a flow node
# # can't open more video captures in the same time
# vcap = cv.VideoCapture(0)
#
# root = Fallback(
#     RightHandAboveCheck(cap=vcap),
#     LeftHandAboveCheck(cap=vcap)
# )
# while True:
#     root.evaluate()