from flow_nodes import NodeStates, Node, Fallback
import cv2 as cv
from yolo_nn import Yolo


class RightHandAboveCheck(Node):

    def __init__(self, *children_nodes, detector):
        super().__init__(*children_nodes)
        self.detector = detector

    def evaluate(self):
        img = self.get_data("img")

        img = self.detector.init_landmarks(img)

        self.detector.get_landmarks(img)

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

    def evaluate(self):
        img = self.get_data("img")

        img = self.detector.init_landmarks(img)

        self.detector.get_landmarks(img)

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


class TrackPerson(Node):
    def __init__(self, *children_nodes):
        super().__init__(*children_nodes)
        self.yolo = None

    def evaluate(self):
        img = self.get_data("img")

        if cv.waitKey(10) & 0xFF == ord('s'):
            init_box = cv.selectROI("Select object for tracking", img, fromCenter=False, showCrosshair=False)
            self.yolo = Yolo(init_box)
            cv.destroyWindow("Select object for tracking")

        if self.yolo is not None:
            img = self.yolo.track(img)
            if self.yolo.tracked_to.box is not None:
                self.parent.set_data("center", self.yolo.tracked_to.centroid)
                tracked_box = self.yolo.tracked_to.box
                cropped_img = self.yolo.crop_im(img, *tracked_box)
                cv.imshow("cropped", cropped_img)  # TODO ?
                cv.waitKey(10)
                self.parent.set_data("cropped_img", cropped_img)
            return NodeStates.SUCCESS
        else:
            cv.imshow('detector', img)
            cv.waitKey(100)
            return NodeStates.FAILURE
