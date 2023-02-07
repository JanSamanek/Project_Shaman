import torch

class Yolo:

    def __init__(self):
        print("  [INF] Initializing Yolo neural network...")
        self.model = torch.hub.load('ultralytics/yolov5','yolov5s')
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
    