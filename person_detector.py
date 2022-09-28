import cv2 as cv
import numpy as np


class Yolo:

    def __init__(self, inp_width=416, inp_height=416,
                 yolo_weights_path='./yolo_nn/yolov3-tiny.weights'
                 , yolo_config_path='./yolo_nn/yolov3-tiny.cfg'):

        with open("yolo_nn/coco.names", 'r') as file:
            self.class_names = file.read().strip().split('\n')

        self.last_layer_names = ""
        self.results = None
        self.inp_width = inp_width
        self.inp_height = inp_height
        self.yolo_weights_path = yolo_weights_path
        self.yolo_config_path = yolo_config_path

    def find_objects(self, image):

        self._setup_yolo_nn(self.yolo_config_path, self.yolo_weights_path)
        self.last_layer_names = self._get_outputs_names()
        blob = self._preprocess_img(image)
        self.results = self._predict(blob)
        img = self._post_process(image, 0.5, 0.4)

        return img

    def _setup_yolo_nn(self, yolo_config, yolo_weights):
        self.yolo_net = cv.dnn.readNetFromDarknet(yolo_config, yolo_weights)
        self.yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def _get_outputs_names(self):
        # Get the names of all the layers in the network
        layer_names = self.yolo_net.getLayerNames()
        # get the names of the output layers, i.e. the layers with unconnected outputs
        return [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

    def _preprocess_img(self, image):
        # preprocesses the image
        # blob is a 4D numpy array object (images, channels, width, height)
        return cv.dnn.blobFromImage(image, 1/255.0, (self.inp_width, self.inp_height), swapRB=True, crop=False)

    def _predict(self, blob):
        # feeds the preprocessed image to the nn
        self.yolo_net.setInput(blob)

        # executes the prediction
        # The outputs object are vectors of length 85
        # 4x the bounding box (center_x, center_y, width, height)
        # 1x box confidence
        # 80x class confidence
        results = self.yolo_net.forward(self.last_layer_names)
        return results

    def _post_process(self, image, conf_threshold, nms_threshold):

        h, w = image.shape[:2]
        GREEN = (0, 255, 0)

        boxes = []
        confidences = []
        classIDs = []

        for result in self.results:
            for detection in result:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > conf_threshold:
                    # adjusting box dimensions to picture dimensions
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = Yolo._non_maximum_suppression(boxes, confidences, conf_threshold, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                # get box dimensions
                (left, top) = (boxes[i][0], boxes[i][1])
                (width, height) = (boxes[i][2], boxes[i][3])
                # draw boxes with confidence
                cv.rectangle(image, (left, top), (left + width, top + height), GREEN, 2)
                text = "{}: {:.4f}".format(self.class_names[classIDs[i]], confidences[i])
                cv.putText(image, text, (left, top - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

        return image

    @staticmethod
    def _non_maximum_suppression(boxes, confidences, conf_threshold, nms_threshold):
        # NON-MAXIMUM-SUPPRESSION
        # removes the boxes that overlap, and have low confidence, until it creates the right one etc.
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        return indices


def main():
    img = cv.imread('horse.jpg')
    yolo = Yolo()
    img = yolo.find_objects(img)
    cv.imshow('OK', img)
    cv.waitKey(0)



if __name__ == '__main__':
    main()
    input()
    cv.destroyAllWindows()
# cap = cv.VideoCapture(0)
#
# while cap.isOpened():
#     # reading the image from video capture
#     _, img = cap.read()

# np.array.shape -> (3, 2) -> první vrstva má v sobě 3 buňky, každá z těch 3 má v sobě pak 2 buňky

# def trackbar(x, image, nn_results):
#     img_shape_x = 416
#     img_shape_y = 416
#
#     confidence = x/100
#     for result in np.vstack(nn_results):
#         # result[4] -> confidence of the box
#         if result[4] > confidence:
#             x, y, w, h = result[:4]
#             p0 = int((x-w/2)*img_shape_x), int((y-h/2)*img_shape_x)
#             p1 = int((x+w/2)*img_shape_y), int((y+h/2)*img_shape_y)
#             cv.rectangle(image, p0, p1, 1, 1)
#     cv.imshow('image', image)
#     cv.waitKey(1)


# trackbar(50, img, results)
