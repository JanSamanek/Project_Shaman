from flow_nodes import NodeStates, Node, Fallback
import cv2 as cv


class RightHandAboveCheck(Node):

    def __init__(self, *children_nodes, detector):
        super().__init__(*children_nodes)
        self.img = None
        self.detector = detector

    def evaluate(self):
        self.img = self.get_data("img")

        self.img = self.detector.init_landmarks(self.img)

        self.detector.get_landmarks(self.img)

        if self.detector.detect_right_hand_above_nose():
            print("Right Hand: Success")
            return NodeStates.SUCCESS
        else:
            print("Right Hand: Failure")
            return NodeStates.FAILURE


class LeftHandAboveCheck(Node):

    def __init__(self, *children_nodes, detector):
        super().__init__(*children_nodes)
        self.detector = detector
        self.img = None

    def evaluate(self):
        self.img = self.get_data("img")

        self.img = self.detector.init_landmarks(self.img)

        self.detector.get_landmarks(self.img)

        if self.detector.detect_left_hand_above_nose():
            print("Left Hand: Success")
            return NodeStates.SUCCESS
        else:
            print("Left Hand: Failure")
            return NodeStates.FAILURE


class CameraCapture(Node):
    def __init__(self, *children_nodes):
        super().__init__(*children_nodes)
        self.cap = cv.VideoCapture(0)

    def evaluate(self):
        success, img = self.cap.read()

        if success:
            self.parent.set_data("img", img)
            return NodeStates.SUCCESS
        else:
            return NodeStates.FAILURE
