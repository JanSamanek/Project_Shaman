from tree import BehaviourTree
from flow_nodes import Fallback, Sequence
from action_nodes import RightHandAboveCheck, LeftHandAboveCheck, CameraCapture
from pose_detector import PoseDetector


class JetsonNano(BehaviourTree):

    def __init__(self):
        self.detector = PoseDetector()
        super().__init__()

    def setup_tree(self):
        self._root = Sequence(
            CameraCapture(),
            Fallback(
                RightHandAboveCheck(detector=self.detector),
                LeftHandAboveCheck(detector=self.detector)
            )
        )


jetson = JetsonNano()
