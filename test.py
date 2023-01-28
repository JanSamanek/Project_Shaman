from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf 

model = load_model('Siamese Network/model/siamese_network.h5')

train_data_dir = r'C:\Users\jands\Market-1501-v15.09.15\bounding_box_train'
file_paths_train = tf.data.Dataset.list_files(train_data_dir + '/*.jpg')

def extract_label(file_path):
    label = tf.strings.split(file_path, '_')
    label = tf.strings.split(label, '\\')
    return int(label[2][1])

def read_and_decode(file_path):
    label = extract_label(file_path)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

print("[INFO] loading data...")

dataset_train = [read_and_decode(file) for file in file_paths_train.take(10)]

import numpy as np
def make_pairs(dataset):
    pairs_list = []
    labels_list = []

    print("[INFO] preparing positive and negative pairs...")

    images = [image for image, label in dataset]
    labels = [label for image, label in dataset]

    unique_labels = np.unique(np.array(labels))

    idxs = [np.where(labels == unique_label)[0] for unique_label in unique_labels]
    for idx_1 in range(len(labels)):
        label = labels[idx_1]
        img_1 = images[idx_1]
        # randomly pick an image that belongs to the *same* class
        for i in range(2):
            idx_2 = np.random.choice(np.where(np.array(labels) == label)[0])
            img_2 = images[idx_2]
            pairs_list.append((img_1, img_2))
            labels_list.append([1])

        # randomly pick an image that does *not* belong to the same class
        for i in range(2):
            idx_2 = np.random.choice(np.where(np.array(labels) != label)[0])
            img_2 = images[idx_2]
            pairs_list.append((img_1, img_2))
            labels_list.append([0])

    return np.array(pairs_list), np.array(labels_list)

train_pair_x, train_pair_y = make_pairs(dataset_train)

img1 = np.expand_dims(train_pair_x[1,0], axis=0)
img2 = np.expand_dims(train_pair_x[1,1], axis=0)


print(model.predict([img1, img2]))