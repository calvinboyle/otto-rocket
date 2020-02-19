import cv2
import imutils

#import video - TODO
#cycle through first 1000 frames skipping a random number of frames between each cycle

#crop and threshold each frame
img = cv2.imread("./test-images/test-image-02.png")
crop_img = img[965:1005, 105:230]
grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
x, threshed = cv2.threshold(grey_crop, 127, 255, cv2.THRESH_BINARY)

#locate digits in the thresholded image
cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

digitCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(crop_img, (x, y), (x+20, y+30), (0,255,0), 1)
    print(x, y, w, h)

#cut digits into individual imgs - TODO

#display each img and ask for user input for classification - TODO

#convert img to array, save array and classification to training dataset - TODO

#display current results
cv2.imshow("resutl", crop_img)
cv2.waitKey(0)
