import socket
import cv2
import numpy as np

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

    def recieve_images(self):
        while True:
            
            # Receive the image size
            size = int.from_bytes(self.client_socket.recv(4), byteorder='big')
            
            # Receive the image data
            data = b''
            while len(data) < size:
                data += self.client_socket.recv(size)

            image = np.frombuffer(data, np.uint8)           # Convert the data to a numpy array
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)   # Decode the image

            cv2.imshow("hello", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()

    
    def close(self):  
        self.client_socket.close()
        self.server_socket.close()

if __name__ == '__main__':
    server = Server()
    server.accept_new_client()
    server.recieve_images()