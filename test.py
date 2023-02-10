from jetcam.csi_camera import CSICamera
from jetbot import ObjectDetector
import cv2 as cv

width, height = 300, 300
camera = CSICamera(width=300, height=300)
camera.running = True

class Tracker():

    def __init__(self):
        self.model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

    def _predict(self, img):
        self.detections = self.model(img)

    def _post_process(self):
        boxes = []
        for detection in self.detections[0]:
            if detection['label'] == 1:
                boxes.append(detection['bbox'])
            
        return boxes
    
def _draw_boxes(image, x_min, y_min, x_max, y_max, color):
    cv.rectangle(image, (int(width * x_min), int(height * y_min)), (int(width * x_max), int(height * y_max)), color, 2)
    return image

tracker = Tracker()
def execute(change):
    img = change['new']
    tracker._predict(img)
    boxes = tracker._post_process()
    for box in boxes:
        _draw_boxes(img, *box, color=(61, 254, 96))
        
    cv.imshow("hello", img)
    cv.waitKey(1)
    
camera.observe(execute, names='value')
