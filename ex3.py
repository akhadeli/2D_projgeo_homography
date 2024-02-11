import numpy as np
import matplotlib.pyplot as plt
import cv2

# Recall
#       --                                                 --
# A_i = |     0.T     ,   -wp_i * x_i.T   ,   yp_i * x_i.T  |
#       | wp_i * x_i.T,       0.T         ,   -xp_i * x_i.T |
#       --                                                 --

DLT = 1

class HomographyEstimator:
    def __init__(self, img_src1, img_src2, mode=DLT, num_points=4):
        self.mode = mode
        self.img_src1 = img_src1
        self.img_src2 = img_src2
        self.num_points = num_points
        self.A = []

    def getPoints(self, frame, num_points):
        # return cv2.selectROI("ROI", frame, fromCenter=False)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        points = np.concatenate((np.array(plt.ginput(num_points)), np.ones((num_points, 1))), axis=1)
        plt.close()
        return points
    
    def start(self):
        if self.mode == DLT:
            self.start_DLT()

    def start_DLT(self):
        img1 = cv2.imread(self.img_src1)
        img2 = cv2.imread(self.img_src2)
        points_1 = self.getPoints(img1, self.num_points)
        points_2 = self.getPoints(img2, self.num_points)

        H = cv2.getPerspectiveTransform(points_1[:, :2].astype(np.float32), points_2[:, :2].astype(np.float32))

        for i in range(self.num_points):
            top = np.hstack((np.zeros(3), -1 * points_2[i][2] * points_1[i], points_2[i][1] * points_1[i]))
            bottom = np.hstack((points_2[i][2] * points_1[i], np.zeros(3), -1 * points_2[i][0] * points_1[i]))
            A_i = np.stack((top, bottom))
            self.A.append(A_i)
        
        self.A = np.vstack(self.A)
        U, S, Vh = np.linalg.svd(self.A)
        h = np.reshape(Vh.T[:, -1], (3, 3))
        print(H)
        print(np.linalg.norm(h))
        print(h)
        
        warped_img1 = cv2.warpPerspective(img1, h, (img1.shape[1], img1.shape[0]))
        warped_img2 = cv2.warpPerspective(img2, h, (img2.shape[1], img2.shape[0]))

        cv2.imshow("warped 1", warped_img1)
        cv2.imshow("warped 2", warped_img2)
        cv2.waitKey(0)


if __name__ == "__main__":
    o = HomographyEstimator('ex3_images/key1.jpg', 'ex3_images/key3.jpg')
    o.start()