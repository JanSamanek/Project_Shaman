import cv2 as cv
from TrackerBase.center_tracker import PersonTracker
from Detector.yolo_nn import Yolo
from Utilities.display import Utility_helper


class Tracker():
    def __init__(self):
        print("[INF] Creating a new tracker...")
        self.yolo = Yolo()
        self.pt = None
        self.tracked_to = None
        self.trackable_objects = None
    
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

    def get_boxes(self, img):
        return self.yolo.predict(img)
    
    def get_to_offset_from_center(self, img_x):
        center = self.tracked_to.centroid if self.tracked_to is not None else None
        center = center if center is not None and img_x > center[0] > 0 else None
        offset = (center[0] - img_x / 2) / (img_x / 2) if center is not None else None
        return offset
    
    def get_to_box(self):
        return self.tracked_to.box if self.tracked_to is not None else None
    
    def set_target(self, center_to):
        self.pt = PersonTracker(center_to)
        self.tracked_to = self.pt.to_dict.get(0, None) if self.pt is not None else None

    def track(self, img, camera_rotation=0, draw_boxes=True, draw_id=True, debug=False):
        boxes = self.yolo.predict(img)
        
        self.trackable_objects = self.pt.update(boxes, camera_rotation)
        self.tracked_to = self.trackable_objects.get(0, None)

        if draw_id:
            for to in self.trackable_objects.values():
                if to.ID == 0 and not to.disappeared_count > 0 and to.centroid is not None:
                    img = Tracker._draw_id(img, to.ID, to.centroid, (18, 13, 212))
                if to.centroid is not None:
                    img = Tracker._draw_id(img, to.ID, to.centroid, (61, 254, 96))
                elif to.disappeared_count > 0 and to.predicted_centroid is not None:
                    img = Tracker._draw_id(img, to.ID, to.predicted_centroid, (253, 63, 28))
            
                if debug:
                    if to.ID == 0 and to.predicted_centroid is not None:
                        img = Tracker._draw_id(img, to.ID, to.predicted_centroid, (253, 63, 28))
                    if to.ID == 0 and to.measured_centroid is not None:
                        img = Tracker._draw_id(img, to.ID, to.measured_centroid, (0,0,0))

        if draw_boxes:
            for box in boxes:
                Tracker._draw_box(img, *box, color=(61, 254, 96))

        return img
    
    
def calculate_center(start_x, start_y, width, height):
    center_x = int((start_x + (start_x + width)) / 2.0)
    center_y = int((start_y + (start_y + height)) / 2.0)
    return center_x, center_y

def create_tracker(img):
    cv.destroyAllWindows()
    to_box = cv.selectROI("Select object for tracking", img, fromCenter=False, showCrosshair=False)
    center = calculate_center(*to_box)
    tracker = Tracker()
    tracker.set_target(center)
    cv.destroyWindow("Select object for tracking")
    return tracker

def main():
    cap = cv.VideoCapture("/home/jan/Project_Shaman/test.mp4")
    previous_time = 0
    tracker = None
    
    while cap.isOpened():
        success, img = cap.read()
        
        if tracker is not None:
            img = tracker.track(img)
            
        previous_time = Utility_helper.display_fps(img, previous_time)
        
        cv.imshow('detector', img)

        if cv.waitKey(1) & 0xFF == ord('s'):
            tracker = create_tracker(img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
     main()
