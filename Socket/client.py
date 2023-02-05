import socket
import cv2
import numpy as np

class Client:
    def __init__(self):   
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object

    def connect_to_server(self, host='10.0.1.103', port=8080):
        # Connect to the server
        print(f"[INF] Trying to connect to ip adress: {host}, port: {port}...")
        self.client_socket.connect((host, port))
        print(f"[INF] Connected to ip adress: {host}, port: {port}...")

    def send_images(self):
        # Create a VideoCapture object
        cap = cv2.VideoCapture(0)

        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()

            result, image = cv2.imencode('.jpg', frame)     # Convert the frame to a JPEG image
            data = image.tobytes()                          # Convert the image to a byte array

            # Send the image size
            self.client_socket.sendall(len(data).to_bytes(4, byteorder='big'))

            # Send the image data
            self.client_socket.sendall(data)

    def disconnect(self):
        print("[INF] Client dissconnecting...")
        self.client_socket.close()
        print("[INF] Client disconnected...")

if __name__ == '__main__':
    client = Client()
    client.connect_to_server()
    client.send_images()
    