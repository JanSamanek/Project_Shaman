import socket
import cv2
import subprocess
import json
from jetcam.csi_camera import CSICamera
from jetbot import Robot

class Client:
    def __init__(self, host='192.168.88.57', port=8080):   
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object
        self.host = host
        self. port = port 

    def connect_to_server(self):
        # Connect to the server
        print(f"[INF] Trying to connect to ip adress: {self.host}, port: {self.port}...")
        self.client_socket.connect((self.host, self.port))
        print(f"[INF] Connected to ip adress: {self.host}, port: {self.port}...")

    def start_streaming(self):
        print(f"[INF] Staming video to {self.host} on port {self.port} ...")
        pipeline = f"gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1280, height=720, framerate=30/1, format=NV12' ! \
                        nvvidconv ! jpegenc ! rtpjpegpay ! udpsink host={self.host} port={self.port}"
        subprocess.Popen(pipeline.split())

    def communicate(self):
        # Create a VideoCapture object
        camera = CSICamera(width=300, height=300)
        camera.running = True

        robot = Robot()
        speed = 0.15
        turn_gain = 0.3
        
        def execute(change):
            img = change['new']
    
            self._send_img(img)
            
            json_data = self._recieve_json()
            center = json_data['center'] 
            stop = json_data['stop']
            
            if center is None:
                pass
                # robot.forward(speed)
            elif stop:
                robot.stop()
                camera.stop()
                self.disconnect()
            else:
                centerx = center[0] - img.shape[0]/2
                robot.set_motors((speed + turn_gain * centerx/100), (speed - turn_gain * centerx/100))

        camera.observe(execute, names='value')
        
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
    
    def disconnect(self):
        print("[INF] Client dissconnecting...")
        self.client_socket.close()
        print("[INF] Client disconnected...")

if __name__ == '__main__':
    client = Client()
    #client.connect_to_server()
    client.start_streaming()
    #client.communicate()
    