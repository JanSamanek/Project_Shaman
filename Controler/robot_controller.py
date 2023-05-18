from Pose.pose_detector import PoseDetector
from Tracker.tracker import Tracker
from Utilities.graphs import plot_and_save_kalman_data

class RobotController():
    def __init__(self):
        self.tracker = Tracker()
        self.pose_detector = PoseDetector()
        self.instruction_img = None

        self.measured = []
        self.predicted = []
        self.kf_centroid = []

    def get_instructions(self, img, camera_rotation=0):
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
            self.instruction_img = self.tracker.track(img, camera_rotation, debug=True)

            if self.tracker.tracked_to is not None:
                offset = self.tracker.get_to_offset_from_center(img.shape[1])
                to_box = self.tracker.get_to_box()

                gestures = self.pose_detector.get_gestures(pose_img, to_box)
                instructions.update(gestures)

                if gestures.get("left_up", False):
                    mot_speed_1, mot_speed_2 = RobotController._get_motor_speed(offset, following=True, turn_gain=0.15, saturation=0.18)
                else:
                    mot_speed_1, mot_speed_2 = RobotController._get_motor_speed(offset, following=False, turn_gain=0.3, saturation=0.15)
                    
                instructions['mot_speed_one'] = mot_speed_1
                instructions['mot_speed_two'] = mot_speed_2

                self.collect_kalman_data(instructions)
               
        return instructions

    def get_instruction_img(self):
        return self.instruction_img
    
    def collect_kalman_data(self, instructions):
        to = self.tracker.tracked_to
        if to.measured_centroid is not None and to.centroid is not None and to.predicted_centroid is not None:
            self.measured.append(self.tracker.tracked_to.measured_centroid) 
            self.kf_centroid.append(self.tracker.tracked_to.centroid)
            self.predicted.append(self.tracker.tracked_to.predicted_centroid)

        if  instructions.get("crossed", False):
            plot_and_save_kalman_data(self.measured, self.kf_centroid, self.predicted)


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

def calculate_center(start_x, start_y, end_x , end_y):
    center_x = int((start_x + end_x) / 2.0)
    center_y = int((start_y + end_y) / 2.0)
    return center_x, center_y

def _sign(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1