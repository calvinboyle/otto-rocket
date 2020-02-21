import numpy as np
import cv2
import os
import random

DATADIR = "./training-data"
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

#reshape array for keras
X = np.array(X).reshape(-1, 30, 20, 1)
