import cv2 as cv
from center_tracker import PersonTracker
from pose_detector import display_fps
from reID_nn import ReID
import tensorflow.keras.backend as K
import torch
# should try to run some of the networks on gpu if possible

class Yolo:

    def __init__(self):
        print("[INF] Initializing Yolo neural network...")
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
    

class Tracker():
    def __init__(self, center_to, ref_image):
        print("[INF] Creating all-in-one tracker ...")
        self.yolo = Yolo()
        self.pt = PersonTracker(center_to)
        self.reid = ReID(ref_image)
        self.trackable_objects = None
        self.tracked_to = None      
    
    @staticmethod
    def _draw_id(image, objectID, centroid, color):
        GREEN = color
        text = "ID {}".format(objectID)
        cv.putText(image, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        cv.circle(image, (int(centroid[0]), int(centroid[1])), 4, GREEN, -1)
        return image
    
    @staticmethod
    def _draw_box(image, x_min, y_min, x_max, y_max, color):
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        return image
    
    @staticmethod
    def crop_im(img, start_x, start_y, end_x, end_y):
        # enlarge the crop
        new_start_y = int(start_y * 0.85)
        new_end_y = int(end_y * 1.1)
        new_start_x = int(start_x * 0.98)
        new_end_x = int(end_x * 1.02)

        if new_end_y > img.shape[0]:
            new_end_y = img.shape[0]
        if new_end_x > img.shape[1]:
            new_end_x = img.shape[1]

        return img[new_start_y:new_end_y, new_start_x:new_end_x]
    
    def track(self, img, reid_on=True, draw_boxes=True, draw_id=True):

        boxes = self.yolo.predict(img)
        
        if reid_on:
            imgs = []
            for box in boxes:
                imgs.append(img[box[1]:box[3], box[0]: box[2]])
            idx = self.reid.identificate(imgs)

            if idx is not None:
                cv.imshow("imgs idx", imgs[idx])
                cv.waitKey(500)
    
        self.trackable_objects = self.pt.update(boxes)
        self.tracked_to = self.trackable_objects.get(0, None)

        if draw_id:
            for to in self.trackable_objects.values():
                if to.ID == 0:
                    img = Tracker._draw_id(img, to.ID, to.centroid, (18, 13, 212))
                elif to.disappeared_count > 0:
                    img = Tracker._draw_id(img, to.ID, to.predicted_centroid, (253, 63, 28))
                else:
                    img = Tracker._draw_id(img, to.ID, to.centroid, (61, 254, 96))

        if draw_boxes:
            for box in boxes:
                Tracker._draw_box(img, *box, color=(61, 254, 96))

        return img
    
    
def calculate_center(start_x, start_y, width, height):
    center_x = int((start_x + (start_x + width)) / 2.0)
    center_y = int((start_y + (start_y + height)) / 2.0)
    return center_x, center_y
    

def main():
    cap = cv.VideoCapture("test.mp4")
    previous_time = 0
    tracker = None
    
    while cap.isOpened():
        _, img = cap.read()

        if tracker is not None:
            img = tracker.track(img, reid_on=False)
            
        previous_time = display_fps(img, previous_time)
        cv.imshow('detector', img)

            
        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.destroyAllWindows()
            to_box = cv.selectROI("Select object for tracking", img, fromCenter=False, showCrosshair=False)
            center = calculate_center(*to_box)
            ref_img = img[to_box[1]:to_box[1] + to_box[3], to_box[0]: to_box[0] + to_box[2]]
            tracker = Tracker(center, ref_img)
            cv.destroyWindow("Select object for tracking")

        # if yolo.tracked_to is not None:
        #     if yolo.tracked_to.box is not None:
        #         to_box = yolo.tracked_to.box
        #         crop_im = yolo.crop_im(orig_img, to_box[0], to_box[1], to_box[2], to_box[3])
        #         cv.imshow('cropped im', crop_im)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
