import torch
import cv2 as cv
from pose_detector import display_fps


def _draw_boxes(image, left, top, width, height):
    GREEN = (0, 255, 0)
    cv.rectangle(image, (left, top), (width, height), GREEN, 2)
    return image


model = torch.hub.load('ultralytics/yolov5', 'custom', path="yolov5/runs/train/exp3/weights/best.pt")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.45

cap = cv.VideoCapture("test.mp4")
previous_time = 0

while cap.isOpened():
    # reading the image from video capture
    _, img = cap.read()

    previous_time = display_fps(img, previous_time)

    results = model(img)

    pd_table = results.pandas().xyxy[0]

    for index, row in pd_table.iterrows():

        img = _draw_boxes(img, int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))

    cv.imshow('detector', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
