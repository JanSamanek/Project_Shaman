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