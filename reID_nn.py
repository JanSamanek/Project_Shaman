from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf 
import cv2 as cv
import numpy as np

class ReID():

    IMG_SIZE = (128, 64)
    CONF = 0.6

    def __init__(self, ref_image):
        self.ref_img = ReID._process_img(ref_image)
        self.reid_model = load_model('Siamese Network/model/siamese_network.h5')
        
    def identificate(self, imgs):
        for idx, img in enumerate(imgs):
            img = ReID._process_img(img)
            prediction = self.reid_model.predict([self.ref_img, img])
            if prediction[0][0] > ReID.CONF:
                return idx ## could be done better, what if another picture has predict bigger than confidence
        else:
            return None

    @staticmethod
    def _process_img(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        tf.image.resize(img, ReID.IMG_SIZE) 
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

    idx = reid.identificate(imgs)

    cv.imshow("ref", img)
    cv.imshow("img 2", imgs[idx])
    cv.waitKey(5000)
