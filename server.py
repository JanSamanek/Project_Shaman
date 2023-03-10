import cv2
import json
from Utilities.display import display_fps
import paho.mqtt.client as mqtt
import subprocess
import time
from Controler.robot_controller import RobotController
    
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
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[INF] Failed to open pipeline ...")
            exit()
        else:
            print(f"[INF] Connected to Gstreamer pipeline on port: {self.gstreamer_port}")
            return cap

    def send_instructions(self, save_video=False):
        previous_time = 0
        robot_controller = RobotController()
        
        cap = self._connect_to_gst_pipeline()

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (1280, 720))

        while cap.isOpened:
            success, img = cap.read()

            if not success:
                print("[ERROR] Failed to fetch image from pipeline ...")
                continue
            
            instructions = robot_controller.get_instructions(img)
            instruction_img = robot_controller.get_instructions_img

            previous_time = display_fps(instruction_img, previous_time)

            if save_video:
                out.write(instruction_img)

            if instructions.get("crossed", False):
                self._publish_json(instructions)
                break
                
            self._publish_json(instructions)
            
            cv2.imshow("*** TRACKING ***", instruction_img)
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
    publisher.send_instructions(save_video=True) 