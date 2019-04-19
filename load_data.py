    
############Load libraries#####################################################
import cv2
import numpy as np
import os
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

###############################################################################
data_dir = './images'
para_data_dir = './images/dog/'
un_data_dir = './images/cat/'
###############################################################################
# declare the number of samples in each category
num_classes = 2
img_rows_orig = 100
img_cols_orig = 100

test_size = .2
seed = 21


def load_data():
    labels = os.listdir(data_dir)
    print(labels)
    assert num_classes == len(labels)

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    c = 0

    X = []
    Y = []
    for label in labels:
        image_names = os.listdir(os.path.join(data_dir, label))
        for image_name in image_names:
            img = cv2.imread(os.path.join(data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array(img)
            X.append(img)
            Y.append(c)

        c += 1

    print('Loading done.')
    Y = np_utils.to_categorical(Y)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = test_size, shuffle = True, random_state = seed)

    return X_train, X_valid, Y_train, Y_valid

def load_resized_data(img_rows, img_cols):
    X_train, X_valid, Y_train, Y_valid = load_data()
    # Resize images
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train])
    X_valid = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_valid])

    X_train = X_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)

    # Data needs to be normalized for it to train correctly
    X_train = np.array(X_train) / 255
    X_valid = np.array(X_valid) / 255

    return X_train, X_valid, Y_train, Y_valid