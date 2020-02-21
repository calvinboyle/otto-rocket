import numpy as np
import cv2
import os
import random
import imutils
from imutils import contours
import xlsxwriter

output = []

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
                digit_key.append([img_array, class_num])
                #digit_key.append(img_array)
            except Exception as e:
                print(path + "failed to open")
                pass

def digit_test(ask_digit):
    for digit in digit_key:
        hold = digit[0]
        digit_score = ask_digit.astype(int) - hold.astype(int)
        #print(digit[1])
        #print(np.sum(abs(digit_score)))
        #print(digit_score)
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
    jump = 1000
    time = count*jump
    
    #crop and threshold each frame
    crop_img = image[965:1000, 105:230]
    grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey_crop, (3,3),0)
    x, threshed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow("value", threshed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #locate digits in the thresholded image
    cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts_sorted = contours.sort_contours(cnts, method="left-to-right")[0]

    value = ''

    for c in cnts_sorted:
        (x, y, w, h) = cv2.boundingRect(c)
        match_score = []
        if w > 10 and h > 10:

            #cut digits into individual imgs
            digit_im = threshed[y:y+30, x:x+20]
            #cv2.imshow("digit", digit_im)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            value += str(digit_test(digit_im))

    output.append([float(value), time])

    count+=1
    cap.set(cv2.CAP_PROP_POS_MSEC, (count*jump))
    success,image = cap.read()


workbook = xlsxwriter.Workbook('alt_data.xlsx')
worksheet = workbook.add_worksheet()

col = 0

for row, data in enumerate(output):
    worksheet.write_row(row, col, data)

workbook.close()





