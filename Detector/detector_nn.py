from jetbot import ObjectDetector

class Detector():

    def __init__(self):
        self.model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

    def predict(self, img):
        self.detections = self.model(img)
        
        boxes = []
        for detection in self.detections[0]:
            if detection['label'] == 1:
                boxes.append(detection['bbox'])
            
        return boxes