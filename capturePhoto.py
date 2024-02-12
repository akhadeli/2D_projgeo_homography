import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('photo', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 115:
        cv2.imwrite('cover.jpg', frame)