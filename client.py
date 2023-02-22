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
        speed = 0.15
        turn_gain = 0.3

        while True:
            json_data = self._recieve_json()
            center = json_data['center'] 
            stop = json_data['stop']
            self._send_mess_acknowledgement()

            if center is None:
                pass
                # robot.forward(speed)
            elif stop:
                robot.stop()
                self.disconnect()
                break
            else:
                centerx = center[0]
                #robot.set_motors((speed + turn_gain * centerx/100), (speed - turn_gain * centerx/100))
        
    def _send_img(self, img):
        result, image = cv2.imencode('.jpg', img)                           # Convert the frame to a JPEG image
        data = image.tobytes()                                              # Convert the image to a byte array
        self.client_socket.sendall(len(data).to_bytes(4, byteorder='big'))  # Send the image size
        self.client_socket.sendall(data)                                    # Send the image data
    
    def _recieve_json(self):
        size = int.from_bytes(self.client_socket.recv(4), byteorder='big')
        json_data = b''
        while len(json_data) < size:
            json_data += self.client_socket.recv(1024)
        return json.loads(json_data)
    
    def _send_mess_acknowledgement(self):
        mess = "recieved"
        mess = mess.encode()
        self.client_socket.sendall(mess.to_bytes(10, byteorder='big'))

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
    