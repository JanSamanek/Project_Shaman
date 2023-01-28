from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf 
import cv2 as cv
import numpy as np
import time

class ReID():

    IMG_SIZE = (128, 64)

    def __init__(self, ref_image):
        self.ref_img = ReID._process_img(ref_image)
        self.reid_model = load_model('Siamese Network/model/siamese_network (1).h5')
        self.conf = 0.7
        self.reid_model.predict([self.ref_img, self.ref_img])  # to reduce predicition time afterwards
        
    def identificate(self, imgs):
        predictions = []
        cv.imshow("ref", np.squeeze(self.ref_img, axis=0))
        for idx, img in enumerate(imgs):
            img = ReID._process_img(img)
            prediction = self.reid_model.predict([self.ref_img, img])
            predictions.append(prediction[0][0])

            # testing
            print("idx: ", idx, ", prediction: ", prediction[0][0])
            cv.imshow(str(idx), np.squeeze(img, axis=0))
            cv.waitKey(1000)

        idx = np.argmax(predictions)
        if predictions[idx] > self.conf:
            return idx
        else:
            return None

    @staticmethod
    def _process_img(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, ReID.IMG_SIZE) 
        img = np.expand_dims(img, axis=0)
        return img

            

if __name__ == '__main__':
    img = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_train\0001_c1s1_001051_00.jpg")

    reid = ReID(img)
    img1 = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_train\0002_c2s1_123141_01.jpg")
    img2 = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_train\0007_c2s3_071002_01.jpg")
    img3 = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_train\0003_c4s6_015641_00.jpg")
    img4 = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_train\0001_c3s1_000551_00.jpg")
    img5 = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_train\0005_c1s1_001351_00.jpg")
    imgs = [img1, img2, img3, img4, img5]

    start = time.perf_counter()
    idx = reid.identificate(imgs)
    end = time.perf_counter()

    total_time = end-start

    print("Elapsed time: ", total_time)
    
    cv.imshow("ref", img)
    cv.imshow("img 2", imgs[idx])
    cv.waitKey(5000)
