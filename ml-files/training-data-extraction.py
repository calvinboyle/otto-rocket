import cv2
import imutils
import os
import numpy as np

os.mkdir("training-data")
os.mkdir("./training-data/0")
os.mkdir("./training-data/1")
os.mkdir("./training-data/2")
os.mkdir("./training-data/3")
os.mkdir("./training-data/4")
os.mkdir("./training-data/5")
os.mkdir("./training-data/6")
os.mkdir("./training-data/7")
os.mkdir("./training-data/8")
os.mkdir("./training-data/9")

data_cnt = [0]*10
filename = 0

#import video
cap = cv2.VideoCapture("./test-images/test-video-01A.mp4")
success,image = cap.read()
count = 0
time = 0

#cycle through first 1000 frames skipping a random number of frames between each cycle
while success:
    jump = np.random.randint(10000, 20000)
    time += jump/1000/60
    print(time)
    cap.set(cv2.CAP_PROP_POS_MSEC, (count*jump))
    success,image = cap.read()

    #crop and threshold each frame
    crop_img = image[965:1005, 105:230]
    grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey_crop, (3,3),0)
    x, threshed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    #locate digits in the thresholded image
    cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    digitCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(crop_img, (x, y), (x+20, y+30), (0,255,0), 1)

        #cut digits into individual imgs - TODO
        digit_im = threshed[y:y+30, x:x+20]

        #display each digit and ask for user imput for classification
        cv2.imshow('digit', digit_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        label = input('enter label: ')
        data_cnt[int(label)] += 1
        print(data_cnt)

        #save each image to folder labeled with classification
        path = os.path.join("./training-data", label, str(filename))
        cv2.imwrite(path+".png", digit_im)
        filename+=1

    count +=1

    

