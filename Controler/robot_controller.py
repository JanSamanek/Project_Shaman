from Pose.pose_detector import PoseDetector
from Tracker.tracker import Tracker, calculate_center

class RobotController():
    def __init__(self):
        self.tracker = Tracker()
        self.pose_detector = PoseDetector()
        self.instruction_img = None

    def get_instructions(self, img):
        instructions = {}
        pose_img = img.copy()

        if self.tracker.pt is None:
            self.instruction_img = img
            boxes = self.tracker.get_boxes(img)
            for box in boxes:
                gestures = self.pose_detector.get_gestures(pose_img, box)
                if gestures.get("right_elevated", False):
                    self.tracker.set_target(calculate_center(*box))
                    break
        else:
            self.instruction_img = self.tracker.track(img)
            offset = self.tracker.get_to_offset_from_center(img.shape[1])
            to_box = self.tracker.get_to_box()

            gestures = self.pose_detector.get_gestures(pose_img, to_box)
            instructions.update(gestures)

            mot_speed_1, mot_speed_2 = self._get_motor_speed(offset)
            instructions['mot_speed_one'] = mot_speed_1
            instructions['mot_speed_two'] = mot_speed_2

        return instructions

    def get_instruction_img(self):
        return self.instruction_img
    
    def _get_motor_speed(self, offset, saturation=0.1, turn_gain=0.3):
        if offset is not None:
            mot_speed_1, mot_speed_2 = (turn_gain * offset, -turn_gain * offset)
            if abs(mot_speed_1) >= saturation or abs(mot_speed_2) >= saturation:
                mot_speed_1, mot_speed_2 = _sign(offset)*saturation, -_sign(offset)*saturation
        else:
            mot_speed_1, mot_speed_2 = None, None
        return mot_speed_1, mot_speed_2


def _sign(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1