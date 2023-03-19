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

        if self.tracker.tracked_to is None:
            self.instruction_img = img
            boxes = self.tracker.get_boxes(img)
            for box in boxes:
                gestures = self.pose_detector.get_gestures(pose_img, box)
                if gestures.get("right_elevated", False):
                    self.tracker.set_target(calculate_center(*box))
                    break
        else:
            self.instruction_img = self.tracker.track(img)

            if self.tracker.tracked_to is not None:
                offset = self.tracker.get_to_offset_from_center(img.shape[1])
                to_box = self.tracker.get_to_box()

                gestures = self.pose_detector.get_gestures(pose_img, to_box)
                instructions.update(gestures)

                if gestures.get("left_up", False):
                    mot_speed_1, mot_speed_2 = RobotController._get_motor_speed(offset, following=True, turn_gain=0.15, saturation=0.15)
                else:
                    mot_speed_1, mot_speed_2 = RobotController._get_motor_speed(offset, following=False, turn_gain=0.3, saturation=0.1)
                    
                instructions['mot_speed_one'] = mot_speed_1
                instructions['mot_speed_two'] = mot_speed_2

        return instructions

    def get_instruction_img(self):
        return self.instruction_img
    
    @staticmethod
    def _get_motor_speed(offset, following, turn_gain, saturation, speed=0.15):
        if offset is not None:
            if following:
                mot_speed_1, mot_speed_2 = (speed + turn_gain * offset, speed - turn_gain * offset)
                if abs(mot_speed_1) >= saturation:
                    mot_speed_1 = _sign(mot_speed_1)*saturation
                if abs(mot_speed_2) >= saturation:
                    mot_speed_2 = _sign(mot_speed_2)*saturation
            else:
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