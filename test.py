# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model

import cv2 as cv
import time

def display_fps(img, previous_time):
    # measuring and displaying fps
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    return previous_time

cap = cv.VideoCapture("test.mp4")
previous_time = 0

def _draw_box(image, x_min, y_min, x_max, y_max, color):
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image
    
while cap.isOpened():
    _, img = cap.read()
        
    previous_time = display_fps(img, previous_time)
        
    results = yolo.predict(img)
    for detection in results[0].boxes.boxes:
        if detection[5] == 0:
            _draw_box(img, int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), (255, 0, 0))
        
    cv.imshow('detector', img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()