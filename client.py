import subprocess
import json
import time
from jetbot import Robot
import paho.mqtt.client as mqtt

class Subscriber():
    def __init__(self, address, topic="jetbot_instructions", port=8080):
        self.address = address
        self.robot = Robot()
        self.client = mqtt.Client()
        self.start_gstreamer()
        self.connect_to_broker(address, port)
        self.client.subscribe(topic)
        self.client.on_message = self.control_robot

    def start_gstreamer(self, gstreamer_port=5000):
        print(f"[INF] Deploying Gstreamer pipeline ...")
        pipeline = f"gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host={self.address} port={gstreamer_port}"
        self.gstreamer_pipeline = subprocess.Popen(pipeline, stdout=subprocess.PIPE, shell=True)
        print(f"[INF] Streaming video to ip adress: {self.address}, port: {gstreamer_port} ...")

    def connect_to_broker(self, address, port):
        self.client.connect(address, port)
        print(f"[INF] Subscriber connected to broker on address: {address}, port: {port} ...")

    def control_robot(self, client, userdata, message):
        global last_time_call
        elapsed_time = time.time() - last_time_call
        print("Time to send and recieve instructions: ", elapsed_time)
        
        SPEED = 0.15
        instructions = json.loads(message.payload.decode())

        mot_speed_1, mot_speed_2 = instructions.get("mot_speed", (None, None))

        # GESTURES
        hands_crossed = instructions.get('crossed', False)
        right_up = instructions.get('right_up', False)
        left_elevated = instructions.get('left_elevated', False)
        left_up = instructions.get('left_up', False)

        if hands_crossed:
            print("[INF] Stopping robot and disconnecting from broker ...")
            self.robot.stop()
            self.stop()
        elif left_up:
            pass
        elif right_up:
            self.robot.set_motors(SPEED, SPEED)
        elif left_elevated:
            self.robot.set_motors(-SPEED, -SPEED)
        elif mot_speed_1 is not None and mot_speed_2 is not None:
            self.robot.set_motors(mot_speed_1, mot_speed_2)
        elif mot_speed_1 is None or mot_speed_2 is None:
            self.robot.stop()            
            
        last_time_call = time.time()

    def run(self):
        self.client.loop_forever()
    
    def stop(self):
        self.gstreamer_pipeline.terminate()
        print("[INF] Gstreamer pipeline disconnected ...")
        self.client.disconnect()
        print("[INF] Subscriber disconnected from broker ...")
        
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Starts client on Jetson Nano")
    parser.add_argument("ip", help="IP adress for the client to connect to")
    args = parser.parse_args()

    last_time_call = time.time()
    subscriber = Subscriber(args.ip)
    subscriber.run()