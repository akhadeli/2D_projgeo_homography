# import numpy as np
# import matplotlib.pyplot as plt
import cv2

save_video_flag = False

cap = cv2.VideoCapture(0)
ret, frame0 = cap.read()
print(frame0.shape)

writer = cv2.VideoWriter('2b_book.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (frame0.shape[1], frame0.shape[0]))

timer = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if save_video_flag:
        writer.write(frame)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if timer == 300:
        save_video_flag = True
    
    if timer == 500:
        break

    print(timer)
    timer += 1
    
    