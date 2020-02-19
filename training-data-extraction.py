import cv2
import imutils
import os

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


#import video - TODO
#cycle through first 1000 frames skipping a random number of frames between each cycle

#crop and threshold each frame
img = cv2.imread("./test-images/test-image-02.png")
crop_img = img[965:1005, 105:230]
grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey_crop, (3,3),0)
x, threshed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

#locate digits in the thresholded image
cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

digitCnts = []
filename = 0

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(crop_img, (x, y), (x+20, y+30), (0,255,0), 1)

    #cut digits into individual imgs - TODO
    digit_im = threshed[y:y+30, x:x+20]

    #display each digit and ask for user imput for classification
    cv2.imshow('digit', digit_im)
    cv2.waitKey(0)
    
    label = input('enter label: ')
    path = os.path.join("./training-data", label, str(filename))

    cv2.imwrite(path+".png", digit_im)
    filename+=1

    #save each image to folder labeled with classification

#display current results
