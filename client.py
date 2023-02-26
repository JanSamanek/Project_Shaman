import socket
import cv2
import subprocess
import json
import time
from jetbot import Robot

class Client:
    def __init__(self, host):   
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object
        self.host = host
        
    def connect_to_server(self, server_port=8080):
        # Connect to the server
        print(f"[INF] Trying to connect to ip adress: {self.host}, port: {server_port} ...")
        self.client_socket.connect((self.host, server_port))
        print(f"[INF] Connected to ip adress: {self.host}, port: {server_port} ...")

    def start_streaming(self, gstreamer_port=5000):
        print(f"[INF] Deploying Gstreamer pipeline ...")
        pipeline = f"gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host={self.host} port={gstreamer_port}"
        self.gstreamer_pipeline = subprocess.Popen(pipeline, stdout=subprocess.PIPE, shell=True)
        print(f"[INF] Streaming video to ip adress: {self.host}, port: {gstreamer_port} ...")

    def communicate(self):
        robot = Robot()

        while True:
        
            json_data = self._recieve_json()

            mot_speed_1, mot_speed_2 = json_data['mot_speed'] 
            stop = json_data['stop']

            if stop:
                robot.stop()
                print("[INF] Stopping robot and disconnecting from server ...")
                self.disconnect()
                break
            elif mot_speed_1 is not None and mot_speed_2 is not None:
                robot.set_motors(mot_speed_1, mot_speed_2)
            elif mot_speed_1 is None or mot_speed_2 is None:
                robot.stop()

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
        self.gstreamer_pipeline.terminate()
        print("[INF] Gstreamer pipeline disconnected ...")
        self.client_socket.close()
        print("[INF] Connection closed ...")

if __name__ == '__main__':
    client = Client("192.168.88.82")
    client.start_streaming()
    client.connect_to_server()
    client.communicate()
    