from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf 
import cv2 as cv

class ReID():

    IMG_SIZE = (128, 64)

    def __init__(self, ref_image):
        self.ref_image = ReID._process_img(ref_image)
        self.reid_model = load_model('Siamese Network/model/siamese_network.h5')
        
    def identificate(imgs):
        for img in imgs:
            self.model.predict()

    @staticmethod
    def _process_img(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        tf.image.resize(img, ReID.IMG_SIZE) 
        return img

            

if __name__ == '__main__':
    img = cv.imread(r"C:\Users\jands\Market-1501-v15.09.15\bounding_box_test\0000_c1s2_002966_05.jpg")
    ReID(img)