import cv2 as cv
cam = cv.VideoCapture(0)
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    istrue, frame = cam.read()
    frame_copy = frame
    if istrue:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor = 1.1, minNeighbors = 7)

        for(x, y, w, h) in faces:
            cv.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
            cut_face = frame[y:y+h, x:x+w]
            cut_face_blur = cv.GaussianBlur(cut_face, (51,51), 0)
            frame[y:y+h, x:x+w] = cut_face_blur


        cv.imshow('live', frame_copy)
        cv.imshow('blur', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()