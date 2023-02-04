import asyncio
import websockets
import json
import cv2
import numpy as np
import base64

def decode_img(img):
    img_decoded = base64.b64decode(img)
    np_arr = np.frombuffer(img_decoded, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


# create handler for each connection
async def handler(websocket, path):
    while True:
        json_message = await websocket.recv()
        img = json.loads(json_message)['img']
        img = decode_img(img)
        
        #cv2.imshow("hello", img)
        #cv2.waitKey(1)
        
        await websocket.send("got your image")
 
 
def start_server():
    start_server = websockets.serve(handler, "localhost", 8000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
    
    
if __name__ == '__main__':
    start_server()