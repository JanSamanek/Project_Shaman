import asyncio
import websockets
import json
import cv2
import base64

def encode_img(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

async def send_image():
    async with websockets.connect("ws://localhost:8000") as websocket:
        while True:
            cap = cv2.VideoCapture(0)
            _, img = cap.read()
            cv2.imshow("hello", img)
            cv2.waitKey(1)
            img_encoded = encode_img(img)
            
            json_message = json.dumps({"img":img_encoded})
            
            await websocket.send(json_message)
            response = await websocket.recv()
            print(response)
 
asyncio.get_event_loop().run_until_complete(send_image())
