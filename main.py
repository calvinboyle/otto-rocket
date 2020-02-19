import cv2
import imutils


#import video - TODO

#crop and threshold each frame
img = cv2.imread("./test-images/test-image-02.png")
crop_img = img[965:1005, 105:230]
grey_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
x, threshed = cv2.threshold(grey_crop, 127, 255, cv2.THRESH_BINARY)

#locate numbers in the thresholded image
cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

digitCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(crop_img, (x, y), (x+20, y+30), (0,255,0), 1)
    print(x, y, w, h)

#sort numbers left to right - TODO

#cut numbers into individual imgs - TODO

#convert img to 20x30 array - TODO

#pass each 20x30 array through pytorch network - TODO

#recombine numeric output into single number - TODO

#export value to file - TODO

#display current results
cv2.imshow("tmp", crop_img)
cv2.waitKey(0)
