import socket
import cv2
import subprocess
import json
from jetbot import Robot

class Client:
    def __init__(self, host, server_port=8080, gstreamer_port=5000):   
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object
        self.host = host
        self.server_port = server_port
        self.gstreamer_port = gstreamer_port 

    def connect_to_server(self):
        # Connect to the server
        print(f"[INF] Trying to connect to ip adress: {self.host}, port: {self.server_port}...")
        self.client_socket.connect((self.host, self.server_port))
        print(f"[INF] Connected to ip adress: {self.host}, port: {self.server_port}...")

    def start_streaming(self):
        print(f"[INF] Deploying Gstreamer pipeline ...")
        pipeline = f"gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host={self.host} port={self.gstreamer_port}"
        print(pipeline)
        self.gstreamer_pipeline = subprocess.Popen(pipeline.split())
        print(f"[INF] Streaming video to ip adress: {self.host}, port: {self.gstreamer_port} ...")

    def communicate(self):
        robot = Robot()
        speed = 0.1
        turn_gain = 0.45

        while True:
            try:
                json_data = self._recieve_json()
            except json.JSONDecodeError as error:
                print("[ERROR] Failed to load json data, emptying buffer ...")
                # empty the buffer
                self.client_socket.recv(0)
                center_x = None
                print("[INF] Seting center to None ...")
            finally:
                center_x = json_data['center_x'] 
                stop = json_data['stop']

                if center_x is not None:
                    print("CENTER: ", center_x)
                    #mot_speed_1 = speed + turn_gain * center_x
                    #mot_speed_2 = speed - turn_gain * center_x
                    mot_speed_1 = turn_gain * center_x
                    mot_speed_2 = -turn_gain * center_x
                    print("MOTOR 1: ", mot_speed_1)
                    print("MOTOR 2: ", mot_speed_2)
                    robot.set_motors(mot_speed_1, mot_speed_2)
                elif center_x is None:
                    robot.stop()
                elif stop:
                    robot.stop()
                    self.disconnect()
                    break

    def _send_img(self, img):
        result, image = cv2.imencode('.jpg', img)                           # Convert the frame to a JPEG image
        data = image.tobytes()                                              # Convert the image to a byte array
        self.client_socket.sendall(len(data).to_bytes(4, byteorder='big'))  # Send the image size
        self.client_socket.sendall(data)                                    # Send the image data
    
    def _recieve_json(self):
        size = int.from_bytes(self.client_socket.recv(4), byteorder='big')
        json_data = self.client_socket.recv(size)
        return json.loads(json_data)

    def disconnect(self):
        # self.gstreamer_pipeline.terminate()
        # print("[INF] Gstreamer pipeline disconnected")
        self.client_socket.close()
        print("[INF] Client disconnected...")

if __name__ == '__main__':
    client = Client("192.168.0.159")
    # client.start_streaming()
    client.connect_to_server()
    client.communicate()
    