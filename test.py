from jetbot import ObjectDetector

class Detector():
    def __init__(self):
        self.model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

    def predict(self, img):
        detections = self.model(img)
        boxes = []
        for det in detections[0]:
            if det['label'] == 1:
                bbox = det['bbox']
                boxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
        return boxes
    
if __name__ == '__main__':
    from jetcam import CSICamera
    import cv2 as cv
    
    def _draw_box(image, x_min, y_min, x_max, y_max, color):
        width = image.shape[1]
        height = image.shape[0]
        cv.rectangle(image, (x_min * width, y_min* height), (x_max * width, y_max * height), color, 2)
        return image
    
    camera = CSICamera(width=300, height=300)
    camera.running = True
    detector = Detector()
    
    def execute(change):
        img = change['new']
        boxes = detector.predict(img)
        
        for box in boxes:
            _draw_box(img, *box, (255,0,0))
            
        cv.imshow('detector', img)
        cv.waitKey(1)
    
    camera.observe(execute, names='value')