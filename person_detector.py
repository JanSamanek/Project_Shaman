import cv2 as cv
from center_tracker import PersonCenterTracker
from pose_detector import display_fps
import torch


class Yolo:

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5',
                                    'custom',
                                    path=r"C:\Users\jands\Project_Shaman\yolov5\runs\train\exp3\weights\best.pt")
        self.model.conf = 0.5
        self.results = None
        self.pd_table = None
        self.centers = None
        self.pt = PersonCenterTracker()

    def _predict(self, img):
        self.results = self.model(img)
        self.pd_table = self.results.pandas().xyxy[0]

    def _post_process(self):
        # process only persons
        boxes = []

        for index, row in self.pd_table.iterrows():
            x_min,  y_min, x_max, y_max = int(row['xmin']),  int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # append box
            boxes.append((x_min, y_min, x_max, y_max))

        return boxes

    def track(self, img, draw_box=True):

        self._predict(img)

        boxes = self._post_process()

        if draw_box:
            for box in boxes:
                Yolo._draw_boxes(img, *box)

        self.centers = self.pt.update(boxes)

        for (objectID, centroid) in self.centers.items():
            img = Yolo._draw_id(img, objectID, centroid)

        return img

    @staticmethod
    def _draw_id(image, objectID, centroid):
        GREEN = (0, 255, 0)
        text = "ID {}".format(objectID)
        cv.putText(image, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        cv.circle(image, (int(centroid[0]), int(centroid[1])), 4, GREEN, -1)
        return image

    @staticmethod
    def _draw_boxes(image, x_min, y_min, x_max, y_max):
        GREEN = (0, 255, 0)
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), GREEN, 2)
        return image

    def get_center(self, ID):
        return self.centers[ID]


def main():
    cap = cv.VideoCapture("test.mp4")
    previous_time = 0
    yolo = Yolo()

    while cap.isOpened():
        # reading the image from video capture
        _, img = cap.read()

        previous_time = display_fps(img, previous_time)

        img = yolo.track(img)

        cv.imshow('detector', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
