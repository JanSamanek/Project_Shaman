import websocket
import base64
import json
import cv2

class WSNode():
    def __init__(self, connection="ws://localhost:8080/"):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(connection, on_message=WSNode.on_message, on_error=WSNode.on_error, on_close=WSNode.on_close)
        self.ws.on_open = WSNode.on_open
        self.ws.run_forever()

    def send(self, data):
        pass

    @staticmethod
    def on_message(ws, message):
        pass
    
    @staticmethod
    def on_error(ws, error):
        print("Error:", error)

    @staticmethod
    def on_close(ws):
        print("Closed websocket connection")

    @staticmethod
    def on_open(ws):
        print("Opened connection")

class Server(WSNode):
    def on_message(self, ws, message):
        json_message = json.loads(message)

class Client(WSNode):
    def send(self, data):
        # Encode image to base64 format
        img_encoded = base64.b64encode(cv2.imencode('.jpg', data)[1]).decode('utf-8')

        json_message = json.dumps({"img": img_encoded})
        self.ws.send(json_message)

    def on_message(self, ws, message):
        json_message = json.loads(message)

