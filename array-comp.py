import numpy as np
import cv2
import os
import random
import imutils
from imutils import contours

DATADIR = "./training-data"
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]
digit_key = []

def create_digit_key():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                #digit_key.append([img_array, class_num])
                digit_key.append(img_array)
            except Exception as e:
                print(path + "failed to open")
                pass

def digit_test(ask_digit):
    for digit in digit_key:
        digit_score = ask_digit-digit[0]
        digit_score = np.sum(abs(digit_score))
        match_score.append(digit_score)
    return np.argmin(match_score)

create_digit_key()

#import video
cap = cv2.VideoCapture("./test-images/test-video-01A.mp4")
success,image = cap.read()
count = 0
time = 0

#cycle through first 1000 frames skipping a random number of frames between each cycle
while success:
    jump = 60000
    cap.set(cv2.CAP_PROP_POS_MSEC, (count*jump))
    success,image = cap.read()

    #crop and threshold each frame
    crop_img = image[960:1005, 105:230]
    grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey_crop, (3,3),0)
    x, threshed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    #locate digits in the thresholded image
    cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts_sorted = contours.sort_contours(cnts, method="left-to-right")[0]

    value = ''

    for c in cnts_sorted:
        match_score = []
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(crop_img, (x, y), (x+20, y+30), (0,255,0), 1)

        #cut digits into individual imgs - TODO
        digit_im = threshed[y:y+30, x:x+20]

        value += str(digit_test(digit_im))

    cv2.imshow('test', threshed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(value)

    count+=1
    





