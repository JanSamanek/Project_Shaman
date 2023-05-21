import subprocess
import json
import time
import paho.mqtt.client as mqtt
import cv2

# BROKER_PORT - port for outside connections is defined in /etc/mosquitto on computer, i overwrote the default config file
BROKER_PORT=8080 

class Client():
    def __init__(self, address):
        self.address = address
        self.client = mqtt.Client()
        self._connect_to_broker(address, BROKER_PORT)

    def _connect_to_broker(self, address, port):
        self.client.connect(address, port)
        print(f"[INF] Client connected to server on address: {address}, port: {port} ...")

    def run(self):
        self.client.loop_forever()
    
    def _stop(self):
        self.client.disconnect()
        print("[INF] Client disconnected from broker ...")


class MqttServer():
    def __init__(self):
        self.mosquitto_server = None

    def start_server(self):
        print(f"[INF] Starting server on port: {BROKER_PORT} ...")
        self.mosquitto_server = subprocess.Popen(f'mosquitto -p {BROKER_PORT}', shell=True)
        time.sleep(2)
    
    def stop_server(self):
        self.mosquitto_server.terminate()
        print("[INF] Server stopped ...")


class Jetbot(Client):
    def __init__(self, address, instruction_topic="jetbot_instructions", rotation_topic="sensor_data"):
        super().__init__(address)
        self.robot = Robot()
        self.mpu = adafruit_mpu6050.MPU6050(busio.I2C(board.SCL_1, board.SDA_1))
        self.rotation_topic = rotation_topic
        self.client.on_message = self.control_robot
        self.client.subscribe(instruction_topic)
        self.start_gstreamer()

    def start_gstreamer(self, gst_port=5000):
        print(f"[INF] Deploying Gstreamer pipeline ...")
        pipeline = f"gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host={self.address} port={gst_port}"
        self.gstreamer_pipeline = subprocess.Popen(pipeline, stdout=subprocess.PIPE, shell=True)
        print(f"[INF] Streaming video to ip adress: {self.address}, port: {gst_port} ...")

    def _send_gyro_data(self):
        z_rot = self.mpu.gyro[2]
        self.client.publish(self.rotation_topic, z_rot, qos=0)

    def control_robot(self, client, userdata, message):    
        SPEED = 0.15
        self._send_gyro_data()
        instructions = json.loads(message.payload.decode())

        mot_speed_1 = instructions.get("mot_speed_one", None)
        mot_speed_2 = instructions.get("mot_speed_two", None)

        # GESTURES
        hands_crossed = instructions.get('crossed', False)
        right_up = instructions.get('right_up', False)
        left_elevated = instructions.get('left_elevated', False)

        if hands_crossed:
            print("[INF] Stopping robot and disconnecting from broker ...")
            self.robot.stop()
            self._stop()
        elif right_up:
            self.robot.set_motors(SPEED, SPEED)
        elif left_elevated:
            self.robot.set_motors(-SPEED, -SPEED)
        elif mot_speed_1 is not None and mot_speed_2 is not None:
            self.robot.set_motors(mot_speed_1, mot_speed_2)
        elif mot_speed_1 is None or mot_speed_2 is None:
            self.robot.stop()            
            
    def _stop(self):
        super()._stop()
        self.gstreamer_pipeline.terminate()
        print("[INF] Gstreamer pipeline disconnected ...")


class InfoPublisher(Client):

    def __init__(self, instruction_topic="jetbot_instructions", rotation_topic="sensor_data", broker_address="localhost"):
        super().__init__(broker_address)
        self.instruction_topic = instruction_topic
        self.client.subscribe(rotation_topic)
        self.client.on_message = self.send_instructions
        self.robot_controller = RobotController()
        self.cap = self._connect_to_gst_pipeline(gst_port=5000)
        self.video_saver = cv2.VideoWriter('testing.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))

    def _connect_to_gst_pipeline(self, gst_port):
        pipeline = f"gst-launch-1.0 udpsrc port={gst_port} ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[INF] Failed to open pipeline ...")
            exit()
        else:
            print(f"[INF] Connected to Gstreamer pipeline on port: {gst_port}")
            return cap

    def send_instructions(self, client, userdata, message):
        camera_rotation = float(message.payload.decode())

        success, img = self.cap.read()

        if not success:
            print("[ERROR] Failed to fetch image from pipeline ...")
            exit()
        
        instructions = self.robot_controller.get_instructions(img, camera_rotation)
        img = self.robot_controller.get_instruction_img()
        
        ######## debug #########
        # Utility_helper.display_camera_rotation(img, camera_rotation)
        # Utility_helper.display_motor_speed(img, instructions.get("mot_speed_one", None), instructions.get("mot_speed_two", None))
        ########################

        if 'previous_time' not in globals():
            global previous_time
            previous_time = 0
        else:
            previous_time = Utility_helper.display_fps(img, previous_time)

        self.video_saver.write(img)

        if instructions.get("crossed", False):
            self._publish_json(instructions)
            self._stop()
            
        self._publish_json(instructions)
        
        cv2.imshow("*** TRACKING ***", img)
        cv2.waitKey(1)

    def _publish_json(self, json_data):
        json_data = json.dumps(json_data)
        self.client.publish(self.instruction_topic, json_data, qos=0)

    def start_communication(self):
        self.client.publish(self.instruction_topic, json.dumps({"data" : None}), qos=0)

    def _stop(self):
        super()._stop()
        self.cap.release()
        print("[INF] Gtsreamer pipeline closed ... ")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_adress", "-ip", help="IP adress for jetbot to connect to", default=None)
    parser.add_argument("--device", "-d", help="Distinguishes which script to execute according to the device it is executed on")
    args = parser.parse_args()


    if args.device == "jetbot":
        from jetbot import Robot
        import board
        import busio
        import adafruit_mpu6050

        jetbot = Jetbot(args.ip_adress)
        jetbot.run()
    else:
        from Controler.robot_controller import RobotController
        from Utilities.display import Utility_helper

        server = MqttServer()
        server.start_server()
        publisher = InfoPublisher()
        publisher.start_communication()
        publisher.run()
        server.stop_server()