from tree import BehaviourTree
from flow_nodes import Fallback, Sequence
from action_nodes import RightHandAboveCheck, LeftHandAboveCheck
import cv2 as cv


class JetsonNano(BehaviourTree):

    def __init__(self):
        self.video_capture = cv.VideoCapture(0)
        super().__init__()

    def setup_tree(self):
        self._root = Fallback(
            Sequence(
                RightHandAboveCheck(cap=self.video_capture)
            ),
            Sequence(
                LeftHandAboveCheck(cap=self.video_capture)
            )
        )


jetson = JetsonNano()
