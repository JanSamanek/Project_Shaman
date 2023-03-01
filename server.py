import socket
import cv2
import numpy as np
import json
from tracker import create_tracker
from Utilities.display_functions import display_fps, display_motor_speed
import time

class Server():
    def __init__(self, server_port=8080, gstreamer_port=5000):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Create a socket object
        self.server_port = server_port
        self.gstreamer_port = gstreamer_port
        self._start_server()
        
    def _start_server(self):
        self.server_socket.bind(('', self.server_port))
        print(f"[INF] Server listening on port: {self.server_port}...")
        self.server_socket.listen(5)
    
    def accept_new_client(self):
        self.client_socket, address = self.server_socket.accept()       # Wait for a client to connect
        print(f"[INF] Client connected from ip adress: {address[0]}...")

    def communicate(self, save_video=False):
        tracker = None
        turn_gain = 0.3
        mot_speed_1, mot_speed_2 = None, None
        offset = None
        previous_time = 0
        
        pipeline = f"gst-launch-1.0 udpsrc port={self.gstreamer_port} ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print("[INF] Failed to open pipeline ...")
            exit()
        else:
            print(f"[INF] Connected to Gstreamer pipeline on port: {self.gstreamer_port}")


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (1280, 720))

        while cap.isOpened:
            success, img = cap.read()
            
            if not success:
                print("[ERROR] Failed to fetch image from pipeline ...")
                continue

            previous_time = display_fps(img, previous_time)
            json_data = {}
            
            if tracker is not None:
                img = tracker.track(img)
                center = tracker.tracked_to.centroid if tracker.tracked_to is not None else None
                center = center if center is not None and img.shape[1] > center[0] > 0 else None        # should rewrite this to be boundaries, what about kalman?
                offset = (center[0] - img.shape[1] / 2) / (img.shape[1] / 2) if center is not None else None

            mot_speed_1, mot_speed_2 = (turn_gain * offset, -turn_gain * offset) if offset is not None else (None, None)

            json_data['mot_speed'] = mot_speed_1, mot_speed_2
            display_motor_speed(img, mot_speed_1, mot_speed_2)

            if save_video:
                out.write(img)
                
            if cv2.waitKey(1) & 0xFF == ord('s'):
                tracker = create_tracker(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                json_data['stop'] = True
                self._send_json(json_data)
                break
            else:
                json_data['stop'] = False
            
            self._send_json(json_data)

            cv2.imshow("*** TRACKING ***", img)
            cv2.waitKey(1)
        
        cap.release()
        print("[INF] Gtsreamer pipeline closed ... ")
        self.close()
            
    def _send_json(self, json_data):
        json_str = json.dumps(json_data)
        data_bytes = json_str.encode()
        
        self.client_socket.sendall(len(data_bytes).to_bytes(4, byteorder='big'))
        self.client_socket.sendall(data_bytes)
        
    def _recieve_img(self):
        size = int.from_bytes(self.client_socket.recv(4), byteorder='big')
        data = b''
        while len(data) < size:
            data += self.client_socket.recv(1024)

        # Decode image
        image = np.frombuffer(data, np.uint8)          
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  
        
        return image

    def close(self):
        print("[INF] Started server terminiation process...")  
        self.client_socket.close()
        self.server_socket.close()
        print("[INF] Server shut down...")  

if __name__ == '__main__':
    server = Server()
    server.accept_new_client()
    server.communicate()