import cv2
import json
from tracker import create_tracker, calculate_center
from Utilities.display import display_fps, display_motor_speed
from Utilities.jetbot_helper import get_motor_speed
import paho.mqtt.client as mqtt
import subprocess
import time
from Pose.pose_detector import PoseDetector
    
class Publisher():
    BROKER_PORT=8080    # port for outside connections is defined in /etc/mosquitto, i overwrote the default config file
    def __init__(self, topic="jetbot_instructions", broker_address="localhost", gstreamer_port=5000):
        self.topic = topic
        self.client = mqtt.Client()
        self.gstreamer_port=gstreamer_port
        self.broker = Publisher._start_broker(Publisher.BROKER_PORT)
        self._connect_to_broker(broker_address, Publisher.BROKER_PORT)

    @staticmethod
    def _start_broker(port):
        print(f"[INF] Starting broker on port: {port} ...")
        mosquitto = subprocess.Popen(f'mosquitto -p {port}', shell=True)
        time.sleep(2)
        return mosquitto

    def _connect_to_broker(self, address, port):
        self.client.connect(address, port)
        print(f"[INF] Publisher connected to broker on address: {address}, port: {port} ...")

    def _connect_to_gst_pipeline(self):
        pipeline = f"gst-launch-1.0 udpsrc port={self.gstreamer_port} ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print("[INF] Failed to open pipeline ...")
            exit()
        else:
            print(f"[INF] Connected to Gstreamer pipeline on port: {self.gstreamer_port}")

    def send_instructions(self, save_video=False):
        previous_time = 0
        tracker, offset = None, None
        pose_detector = PoseDetector()

        self._connect_to_gst_pipeline()

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (1280, 720))

        while self.cap.isOpened:
            success, img = self.cap.read()

            if not success:
                print("[ERROR] Failed to fetch image from pipeline ...")
                continue
            
            instructions = {}
            gestures = {}

            if tracker is not None:
                pose_img = img.copy()

                img = tracker.track(img)
                offset = tracker.get_to_offset_from_center(img.shape[1])
                to_box = tracker.get_to_box()

                if to_box is not None:
                    gestures = pose_detector.get_gestures(pose_img, to_box)
                    instructions.update(gestures)
                elif tracker.tracked_to is None:
                    boxes = tracker.get_boxes(img)
                    for box in boxes:
                        gestures = pose_detector.get_gestures(pose_img, box)
                        if gestures.get("right_up", False):
                            tracker.update_target(calculate_center(*box))
                            break
            
            mot_speed_1, mot_speed_2 = get_motor_speed(offset)

            instructions['mot_speed'] = mot_speed_1, mot_speed_2

            previous_time = display_fps(img, previous_time)
            display_motor_speed(img, mot_speed_1, mot_speed_2)

            if save_video:
                out.write(img)
                
            if cv2.waitKey(1) & 0xFF == ord('s'):
                tracker = create_tracker(img)

            if gestures.get("crossed", False):
                self._publish_json(instructions)
                break
                
            self._publish_json(instructions)
            
            cv2.imshow("*** TRACKING ***", img)
            cv2.waitKey(1)
        
        self.cap.release()
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
    publisher.send_instructions(save_video=True) 