import numpy as np
import matplotlib.pyplot as plt
import cv2

# Recall
#       --                                                 --
# A_i = |     0.T     ,   -wp_i * x_i.T   ,   yp_i * x_i.T  |
#       | wp_i * x_i.T,       0.T         ,   -xp_i * x_i.T |
#       --                                                 --

DLT = 1
L1_DLT = 2
NORM_DLT = 3

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
        elif self.mode == L1_DLT:
            self.start_L1_DLT()
        elif self.mode == NORM_DLT:
            self.start_norm_dlt()

    def start_DLT(self):
        img1 = cv2.imread(self.img_src1)
        img2 = cv2.imread(self.img_src2)
        points_1 = self.getPoints(img1, self.num_points)
        points_2 = self.getPoints(img2, self.num_points)

        for i in range(self.num_points):
            top = np.hstack((np.zeros(3), -1 * points_2[i][2] * points_1[i], points_2[i][1] * points_1[i]))
            bottom = np.hstack((points_2[i][2] * points_1[i], np.zeros(3), -1 * points_2[i][0] * points_1[i]))
            A_i = np.stack((top, bottom))
            self.A.append(A_i)
        
        self.A = np.vstack(self.A)
        U, S, Vh = np.linalg.svd(self.A)
        h = np.reshape(Vh.T[:, -1], (3, 3))
        
        warped_img1 = cv2.warpPerspective(img1, h, (img1.shape[1], img1.shape[0]))
        warped_img2 = cv2.warpPerspective(img2, h, (img2.shape[1], img2.shape[0]))

        cv2.imshow("warped 1", warped_img1)
        cv2.imshow("warped 2", warped_img2)
        cv2.waitKey(0)

    def start_L1_DLT(self):
        img1 = cv2.imread(self.img_src1)
        img2 = cv2.imread(self.img_src2)
        points_1 = self.getPoints(img1, self.num_points)
        points_2 = self.getPoints(img2, self.num_points)

        eta = np.ones(2*self.num_points)
        diag_eta = np.diag(1/np.sqrt(eta))

        for i in range(self.num_points):
            top = np.hstack((np.zeros(3), -1 * points_2[i][2] * points_1[i], points_2[i][1] * points_1[i]))
            bottom = np.hstack((points_2[i][2] * points_1[i], np.zeros(3), -1 * points_2[i][0] * points_1[i]))
            A_i = np.stack((top, bottom))
            self.A.append(A_i)
        
        self.A = np.vstack(self.A)
        self.A = np.dot(diag_eta, self.A)

        U, S, Vh = np.linalg.svd(self.A)
        h = Vh.T[:, -1]

        warped_img1 = cv2.warpPerspective(img1, np.reshape(Vh.T[:, -1], (3, 3)), (img1.shape[1], img1.shape[0]))
        # warped_img2 = cv2.warpPerspective(img2, np.reshape(Vh.T[:, -1], (3, 3)), (img2.shape[1], img2.shape[0]))

        cv2.imshow("warped 1", warped_img1)
        # cv2.imshow("warped 2", warped_img2)

        l1_min = 0.5 * (np.ones(self.num_points*2) @ eta + np.linalg.norm(np.dot(self.A, h), 2)**2)
        change_l1 = 100

        start_norm = np.linalg.norm(self.A @ h)
        change_norm = 100
        steps = 1

        while (change_l1 > 1.01 or change_l1 < 0.99) or (change_norm > 1.01 or change_norm < 0.99):
            steps+=1
            
            eta = np.abs(self.A @ h)
            
            diag_eta = np.diag(1/np.sqrt(eta))

            self.A = []

            for i in range(self.num_points):
                top = np.hstack((np.zeros(3), -1 * points_2[i][2] * points_1[i], points_2[i][1] * points_1[i]))
                bottom = np.hstack((points_2[i][2] * points_1[i], np.zeros(3), -1 * points_2[i][0] * points_1[i]))
                A_i = np.stack((top, bottom))
                self.A.append(A_i)
            
            self.A = np.vstack(self.A)
            self.A = np.dot(diag_eta, self.A)

            U, S, Vh = np.linalg.svd(self.A)
            h = np.abs(Vh.T[:, -1])

            l1_min_new = 0.5 * (np.ones(self.num_points*2) @ eta + np.linalg.norm(np.dot(self.A, h), 2)**2)
            change_l1 = l1_min/l1_min_new
            l1_min = l1_min_new

            change_norm = np.linalg.norm(self.A @ h)/start_norm
            start_norm = np.linalg.norm(self.A @ h)
        
        print(f"Conv in {steps} steps")
        warped_img1_l1 = cv2.warpPerspective(img1, np.reshape(Vh.T[:, -1], (3, 3)), (img1.shape[1], img1.shape[0]))
        # warped_img2_l1 = cv2.warpPerspective(img2, np.reshape(Vh.T[:, -1], (3, 3)), (img2.shape[1], img2.shape[0]))
        cv2.imshow("warped 1 l1", warped_img1_l1)
        # cv2.imshow("warped 2 l1", warped_img2_l1)

        cv2.waitKey(0)
    
    def start_norm_dlt(self):
        img1 = cv2.imread(self.img_src1)
        img2 = cv2.imread(self.img_src2)
        points_1 = self.getPoints(img1, self.num_points)
        points_2 = self.getPoints(img2, self.num_points)
        
        h, w, _ = img1.shape
        hp, wp, _ = img2.shape

        T_norm = np.linalg.inv(np.array([[w+h, 0, w/2],
                           [0, w+h, h/2],
                           [0, 0, 1]]))
        Tp_norm = np.linalg.inv(np.array([[wp+hp, 0, wp/2],
                           [0, wp+hp, hp/2],
                           [0, 0, 1]]))
        
        points_1_norml = (T_norm @ points_1.T).T
        points_2_norml = (Tp_norm @ points_2.T).T

        for i in range(self.num_points):
            top = np.hstack((np.zeros(3), -1 * points_2_norml[i][2] * points_1_norml[i], points_2_norml[i][1] * points_1_norml[i]))
            bottom = np.hstack((points_2_norml[i][2] * points_1_norml[i], np.zeros(3), -1 * points_2_norml[i][0] * points_1_norml[i]))
            A_i = np.stack((top, bottom))
            self.A.append(A_i)
        
        self.A = np.vstack(self.A)
        U, S, Vh = np.linalg.svd(self.A)
        h = np.reshape(Vh.T[:, -1], (3, 3))

        H = (np.linalg.inv(Tp_norm) @ h) @ T_norm
        
        warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
        warped_img2 = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

        cv2.imshow("warped 1", warped_img1)
        cv2.imshow("warped 2", warped_img2)
        cv2.waitKey(0)
        


if __name__ == "__main__":
    o = HomographyEstimator('ex3_images/key1.jpg', 'ex3_images/key3.jpg', mode=L1_DLT, num_points=5)
    o.start()