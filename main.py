import cv2 as cv
import numpy as np
img = cv.imread('cats_vs_dogs.png')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)
copy_img = img.copy()
shrink_img = cv.resize(img, (400,400))
cut_img = img[100:500,150:450]
blur_img = cv.GaussianBlur(img, (15,15), 0)
edges = cv.Canny(blur_img, 50, 200)
cv.imshow('img', img)
cv.imshow('grayimg', gray_img)
cv.imshow('red rectangle',cv.rectangle(img, (50,50), (650,550), (0, 0, 255), 3))
cv.imshow('text', cv.putText(img,'Cat', (50,40), cv.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2))
cv.imshow('shrinked img', shrink_img)
cv.imshow('cut img', cut_img)
cv.imshow('blur img', blur_img)
cv.imshow('edges', edges)
cv.waitKey(0)
cv.destroyAllWindows()

#cam = cv.VideoCapture(0)
#while True:
#    istrue, frame = cam.read()
#    if istrue:
#       cv.imshow('live', frame)
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break
#cam.release()
#cv.destroyAllWindows()

#cam = cv.VideoCapture(0)
#while True:
#    istrue, frame = cam.read()
#    if istrue:
#        cv.imshow('live', cv.Canny(frame, 100, 200))
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break
#cam.release()
#cv.destroyAllWindows()

#cam = cv.VideoCapture(0)
#while True:
#    istrue, frame = cam.read()
#    if istrue:
#        center_x = frame.shape[1] // 2
#        center_y = frame.shape[0] // 2
#        cv.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 0, 255), 2)
#        cv.putText(frame,'target',(center_x - 40, center_y - 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#        cv.imshow('live', frame)
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break
#cam.release()
#cv.destroyAllWindows()

cam = cv.VideoCapture(0)
while True:
    istrue, frame = cam.read()
    if istrue:
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        min_bound = np.array([150, 75, 0])
        max_bound = np.array([180, 255, 255])
        mask = cv.inRange(hsv_frame, min_bound, max_bound)
        kernel = np.ones((3,3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        colored_result = cv.bitwise_and(frame, frame, mask = mask)
        cv.imshow('mask', mask)
        cv.imshow('result', colored_result)

        contours, hierarchy = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        cv.drawContours(
            colored_result,
            contours,
            -1,
            (0, 255, 0),
            1
        )
        cv.imshow("contours", colored_result)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()

