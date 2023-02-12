import cv2 as cv
from TrackerBase.center_tracker import PersonTracker
from test import Detector
import time


class Tracker():
    def __init__(self, center_to):
        print("[INF] Creating all-in-one tracker...")
        self.detector = ObjectDetector()
        self.pt = PersonTracker(center_to)
    
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
        width = image.shape[1]
        height = image.shape[0]
        cv.rectangle(image, (x_min * width, y_min* height), (x_max * width, y_max * height), color, 2)
        return image

    def track(self, img, draw_boxes=True, draw_id=True):
        boxes = self.detector.predict(img)
        
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


def create_tracker(img):
    cv.destroyAllWindows()
    to_box = cv.selectROI("Select object for tracking", img, fromCenter=False, showCrosshair=False)
    center = calculate_center(*to_box)
    tracker = Tracker(center)
    cv.destroyWindow("Select object for tracking")
    return tracker
        
def display_fps(img, previous_time):
    # measuring and displaying fps
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    return previous_time

def main():
    cap = cv.VideoCapture("test.mp4")
    previous_time = 0
    tracker = None
    
    while cap.isOpened():
        _, img = cap.read()

        if tracker is not None:
            img = tracker.track(img)
            
        previous_time = display_fps(img, previous_time)
        cv.imshow('detector', img)

        if cv.waitKey(1) & 0xFF == ord('s'):
            tracker = create_tracker(img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
     main()
