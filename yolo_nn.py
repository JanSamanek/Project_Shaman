import cv2 as cv
from center_tracker import PersonTracker
from pose_detector import display_fps
import torch


class Yolo:

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5','yolov5s')
        self.model.conf = 0.5
        self.results = None
        self.pd_table = None
        self.trackable_objects = None
        self.tracked_to = None
        self.pt = None

    def init_tracking(self, box_to):
        centre_to = Yolo._calculate_center(*box_to)
        self.pt = PersonTracker(centre_to)

    def _predict(self, img):
        self.results = self.model(img)
        self.pd_table = self.results.pandas().xyxy[0]
        self.pd_table = self.pd_table.loc[self.pd_table['name'] == 'person']

    def _post_process(self):
        # process only persons
        boxes = []

        for index, row in self.pd_table.iterrows():
            x_min,  y_min, x_max, y_max = int(row['xmin']),  int(row['ymin']), int(row['xmax']), int(row['ymax'])
            boxes.append((x_min, y_min, x_max, y_max))

        return boxes

    def track(self, img):

        self._predict(img)

        boxes = self._post_process()

        for box in boxes:
            Yolo._draw_boxes(img, *box, color=(61, 254, 96))

        if self.pt is not None:
            self.trackable_objects = self.pt.update(boxes)
            self.tracked_to = self.trackable_objects.get(0, None)

            for to in self.trackable_objects.values():
                if to.ID == 0:
                    img = Yolo._draw_id(img, to.ID, to.centroid, (18, 13, 212))
                elif to.disappeared_count > 0:
                    img = Yolo._draw_id(img, to.ID, to.predicted_centroid, (253, 63, 28))
                else:
                    img = Yolo._draw_id(img, to.ID, to.centroid, (61, 254, 96))

        return img

    @staticmethod
    def crop_im(img, start_x, start_y, end_x, end_y):
        # enlarge the crop by 20%
        new_start_y = int(start_y * 0.85)
        new_end_y = int(end_y * 1.1)
        new_start_x = int(start_x * 0.98)
        new_end_x = int(end_x * 1.02)

        if new_end_y > img.shape[0]:
            new_end_y = img.shape[0]
        if new_end_x > img.shape[1]:
            new_end_x = img.shape[1]

        return img[new_start_y:new_end_y, new_start_x:new_end_x]

    @staticmethod
    def _draw_id(image, objectID, centroid, color):
        GREEN = color
        text = "ID {}".format(objectID)
        cv.putText(image, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        cv.circle(image, (int(centroid[0]), int(centroid[1])), 4, GREEN, -1)
        return image

    @staticmethod
    def _draw_boxes(image, x_min, y_min, x_max, y_max, color):
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        return image

    @staticmethod
    def _calculate_center(start_x, start_y, width, height):
        center_x = int((start_x + (start_x + width)) / 2.0)
        center_y = int((start_y + (start_y + height)) / 2.0)
        return center_x, center_y


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

        if yolo.tracked_to is not None:
            if yolo.tracked_to.box is not None:
                to_box = yolo.tracked_to.box
                crop_im = yolo.crop_im(img, to_box[0], to_box[1], to_box[2], to_box[3])
                cv.imshow('cropped im', crop_im)

        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.destroyAllWindows()
            tb_box = cv.selectROI("Select object for tracking", img, fromCenter=False, showCrosshair=False)
            yolo.init_tracking(tb_box)
            cv.destroyWindow("Select object for tracking")
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def _draw_boxes(image, x_min, y_min, x_max, y_max):
    GREEN = (0, 255, 0)
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), GREEN, 2)
    return image


def yolo_test():
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


if __name__ == '__main__':
    main()
