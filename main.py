import cv2
from random import randint

trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#img = cv2.imread('C:\VS Code\Py PlayGround\Face Detection\GP.jpg')
webcam = cv2.VideoCapture(0)

while True:

    succesful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randint(0, 255),
                                                  randint(0, 255), randint(0, 255)), 2)

    cv2.imshow('KK', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
