import cv2
import json
from tracker import create_tracker
from Utilities.display_functions import display_fps, display_motor_speed
import paho.mqtt.client as mqtt
import subprocess
import time
from Pose.pose_detector import PoseDetector
from test2 import get_camera_shift
    
class Publisher():
    BROKER_PORT=8080    # port for outside connections is defined in /etc/mosquitto, i overwrote the default config file
    def __init__(self, topic="jetbot_instructions", broker_address="localhost", gstreamer_port=5000):
        self.topic = topic
        self.client = mqtt.Client()
        self.gstreamer_port=gstreamer_port
        self.broker = Publisher._start_broker(broker_address,Publisher.BROKER_PORT)
        self._connect_to_broker(broker_address, Publisher.BROKER_PORT)

    @staticmethod
    def _start_broker(address,port):
        print(f"[INF] Starting broker on port: {port} ...")
        mosquitto = subprocess.Popen(f'mosquitto -p {port}', shell=True)
        time.sleep(2)
        return mosquitto

    def _connect_to_broker(self, address, port):
        self.client.connect(address, port)
        print(f"[INF] Publisher connected to broker on address: {address}, port: {port} ...")

    def send_instructions(self, save_video=False):
        TURN_GAIN = 0.35
        previous_time = 0
        tracker, mot_speed_1, mot_speed_2, offset = None, None, None, None
        pose_detector = PoseDetector()

        pipeline = f"gst-launch-1.0 udpsrc port={self.gstreamer_port} ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[INF] Failed to open pipeline ...")
            exit()
        else:
            success, img1 = cap.read()
            print(f"[INF] Connected to Gstreamer pipeline on port: {self.gstreamer_port}")

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (1280, 720))

        while cap.isOpened:
            success, img2 = cap.read()
            # dx = get_camera_shift(img1, img2)
            # if dx < 0:
            #     print("LEFT: ", dx)
            # else:
            #     print("RIGHT: ", dx)
            
            # img1 = img2

            if not success:
                print("[ERROR] Failed to fetch image from pipeline ...")
                continue
            
            json_data = {}
            json_data['right_hand_gest'] = False
            
            if tracker is not None:
                pose_img = img2.copy()

                img2 = tracker.track(img2)
                center = tracker.tracked_to.centroid if tracker.tracked_to is not None else None
                center = center if center is not None and img2.shape[1] > center[0] > 0 else None        # should rewrite this to be boundaries, what about kalman?
                offset = (center[0] - img2.shape[1] / 2) / (img2.shape[1] / 2) if center is not None else None

                to_box = tracker.tracked_to.box if tracker.tracked_to is not None else None

                if to_box is not None:
                    pose_img = pose_detector.get_landmarks(pose_img, box=to_box)
                    left_hand_up = pose_detector.detect_left_hand_above_nose()
                    right_hand_up = pose_detector.detect_right_hand_above_nose()

                    json_data['right_hand_gest'] = right_hand_up
                    
                    if left_hand_up and right_hand_up:
                        print("[INF] Both hands above nose detected")
                    elif left_hand_up:
                        print("[INF] Left hand gesture above nose detected")
                    elif right_hand_up:
                        print("[INF] Right hand gesture above nose detected")

            mot_speed_1, mot_speed_2 = (TURN_GAIN * offset, -TURN_GAIN * offset) if offset is not None else (None, None)

            json_data['mot_speed'] = mot_speed_1, mot_speed_2

            previous_time = display_fps(img2, previous_time)
            display_motor_speed(img2, mot_speed_1, mot_speed_2)

            if save_video:
                out.write(img2)
                
            if cv2.waitKey(1) & 0xFF == ord('s'):
                tracker = create_tracker(img2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                json_data['stop'] = True
                self._publish_json(json_data)
                break
            else:
                json_data['stop'] = False
                
            self._publish_json(json_data)
            
            cv2.imshow("*** TRACKING ***", img2)
            cv2.waitKey(1)
        
        cap.release()
        print("[INF] Gtsreamer pipeline closed ... ")
        self._terminate()

    def _publish_json(self, json_data):
        json_data = json.dumps(json_data)
        self.client.publish(self.topic, json_data, qos=0)

    def _terminate(self):
        self.client.disconnect()
        print("[INF] Disconected publisher from broker ...")
        self.broker.terminate()
        print("[INF] Terminated broker ...")

if __name__ == '__main__':
    publisher = Publisher()
    publisher.send_instructions() 