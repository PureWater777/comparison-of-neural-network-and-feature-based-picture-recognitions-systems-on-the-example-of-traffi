import os

import cv2
from cv2 import imread
import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model('model.h5')

model.summary()

def resize_cv(img):
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)


image_name = "00006.ppm"
img_path = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/" + image_name
directory = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/"
csv_file = pd.read_csv('GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/GT-final_test.test.csv', sep=';')
images = []
for row in csv_file.iterrows():
    if row[1].Filename != image_name:
        continue
    img_path = os.path.join(directory, row[1].Filename)
    img = imread(img_path)
    a = row[1]['Roi.X1']
    b = row[1]['Roi.X2']
    c = row[1]['Roi.Y1']
    d = row[1]['Roi.Y2']
    img = img[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
    img = resize_cv(img)
    images.append(img)
stack_img = np.stack(images)
pred = model.predict([stack_img, ])
print(pred)
max_index_col = np.argmax(pred[0], axis=0)
print(max_index_col)