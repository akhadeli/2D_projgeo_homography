import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
import cv2

source = 'test.jpg'
image = plt.imread(source)
frame = cv2.imread(source)

fig, ax = plt.subplots()
plt.imshow(image)
points = np.concatenate((np.array(plt.ginput(4)), np.ones((4, 1))), axis=1)
print(points)

# Points selected in following order:
# 0: bottom-left
# 1: top-right
# 2: bottom-right
# 3: top-left
l1 = np.cross(points[0], points[1])
l2 = np.cross(points[2], points[3])
l3 = np.cross(points[0], points[2])
l4 = np.cross(points[1], points[3])

cv2.line(frame, (np.int16(points[0][0]), np.int16(points[0][1])), (np.int16(points[1][0]), np.int16(points[1][1])), (255, 0, 0), 1)
cv2.line(frame, (np.int16(points[2][0]), np.int16(points[2][1])), (np.int16(points[3][0]), np.int16(points[3][1])), (255, 0, 0), 1)

# cv2.line(frame, (np.int16(points[0][0]), np.int16(points[0][1])), (np.int16(points[2][0]), np.int16(points[2][1])), (255, 0, 0), 1)
# cv2.line(frame, (np.int16(points[1][0]), np.int16(points[1][1])), (np.int16(points[3][0]), np.int16(points[3][1])), (255, 0, 0), 1)

print(l1, l2)
print(l3, l4)

x_m = np.cross(l1, l2)
cv2.circle(frame, (np.int16(x_m[0]/x_m[2]), np.int16(x_m[1]/x_m[2])), 3, (0, 0, 255), 3)

x_inf = np.cross(l3, l4)
cv2.circle(frame, (np.int16(x_inf[0]/x_inf[2]), np.int16(x_inf[1]/x_inf[2])), 3, (0, 0, 255), 3)
print(np.int16(x_inf[0]/x_inf[2]), np.int16(x_inf[1]/x_inf[2]))

lm = np.cross(x_inf, x_m)

cv2.line(frame, (np.int16(x_inf[0]/x_inf[2]), np.int16(x_inf[1]/x_inf[2])), (np.int16(x_m[0]/x_m[2]), np.int16(x_m[1]/x_m[2])), (0, 0, 255), 2)

cv2.imshow("frame", frame)
cv2.waitKey(0)