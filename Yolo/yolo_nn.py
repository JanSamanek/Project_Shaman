import torch

class Yolo:

    def __init__(self):
        print("  [INF] Initializing Yolo neural network...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5','yolov5s')
        self.model.to(device)
        self.model.conf = 0.5

    def predict(self, img):
        boxes = []
        results = self.model(img)
        pd_table = results.pandas().xyxy[0]
        pd_table = pd_table.loc[pd_table['name'] == 'person']
        
        for index, row in pd_table.iterrows():
            x_min,  y_min, x_max, y_max = int(row['xmin']),  int(row['ymin']), int(row['xmax']), int(row['ymax'])
            boxes.append((x_min, y_min, x_max, y_max))
            
        return boxes
    
if __name__ == '__main__':
    import cv2 as cv
    import time
    
    def display_fps(img, previous_time):
        # measuring and displaying fps
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time
        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        return previous_time

    def _draw_box(image, x_min, y_min, x_max, y_max, color):
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        return image
    
    
    cap = cv.VideoCapture("test.mp4")
    previous_time = 0
    yolo = Yolo()
    
    while cap.isOpened():
        _, img = cap.read()
            
        previous_time = display_fps(img, previous_time)
        boxes = yolo.predict(img)
        for box in boxes:
             _draw_box(img, *box, (255, 0, 0))
        
        cv.imshow('detector', img)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()