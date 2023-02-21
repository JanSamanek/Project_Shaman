import socket
import cv2
import numpy as np
import json
from tracker import create_tracker, display_fps

class Server():
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # Create a socket object
        self._start_server()
        
    def _start_server(self, port=8080):
        host = socket.gethostname()
        ip_adress = socket.gethostbyname(host)
        self.server_socket.bind((host, port))
        print(f"[INF] Server listening on ip adress: {ip_adress}, port: {port}...")
        self.server_socket.listen(5)
    
    def accept_new_client(self):
        # Wait for a client to connect
        self.client_socket, address = self.server_socket.accept()
        print(f"[INF] Client connected from ip adress: {address[0]}...")

    def communicate(self):
        tracker = None
        center = None
        previous_time = 0
        
        while True:
            img = self._recieve_img()
            previous_time = display_fps(img, previous_time)
            json_data = {}
            
            if tracker is not None:
                img = tracker.track(img)
                center = tracker.tracked_to.centroid if tracker.tracked_to is not None else None
                
            if cv2.waitKey(1) & 0xFF == ord('s'):
                tracker = create_tracker(img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                json_data['stop'] = True
            else:
                json_data['stop'] = False
                
            json_data['center'] = center
            
            self._send_json(json_data)
            
            cv2.imshow("*** TRACKING ***", img)
            cv2.waitKey(1)
            
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