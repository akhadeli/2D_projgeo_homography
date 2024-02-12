import numpy as np
import matplotlib.pyplot as plt
import cv2

class AugmentedReality:
    def __init__(self, cap_source, patch_size=85, num_points=4):
        self.source = cap_source
        self.num_points = num_points
        self.patch_size = patch_size
        self.trackers = []
        self.A = []

    def getPoints(self, frame, num_points):
        # return cv2.selectROI("ROI", frame, fromCenter=False)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        points = np.concatenate((np.array(plt.ginput(num_points)), np.ones((num_points, 1))), axis=1)
        plt.close()
        return points
    
    def getRect(self, frame):
        return cv2.selectROI("ROI", frame, fromCenter=False)
    
    def initTrackers(self, frame, points, num_trackers):
        # x y 1
        for i in range(num_trackers):
            bbox = (np.int32(points[i][0] - (self.patch_size-1)/2), np.int32(points[i][1] - (self.patch_size-1)/2), np.int32(self.patch_size), np.int32(self.patch_size))
            self.trackers.append(cv2.TrackerKCF_create())
            self.trackers[i].init(frame, bbox)

    def getTrackerCenter(self, bbox):
        return np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, 1], dtype=np.int32)
    
    def drawCircle(self, frame, center, radius=2, color=(0, 0, 255), thickness=1):
        cv2.circle(frame, center, radius, color, thickness)
    
    def drawLine(self, frame, p1, p2, color=(255, 0, 0), thickness=2):
        cv2.line(frame, p1, p2, color, thickness)

    def start(self):
        cover = cv2.imread('cover.jpg')
        # points_roi = self.getRect(cover)
        # points_cover = np.array([[points_roi[0], points_roi[1], 1],
        #                          [points_roi[0]+points_roi[2], points_roi[1], 1],
        #                          [points_roi[0], points_roi[1]+points_roi[3], 1],
        #                          [points_roi[0]+points_roi[2], points_roi[1]+points_roi[3], 1]], dtype=np.float32)
        points_cover = self.getPoints(cover, 4)
        cheight, cwidth, _ = cover.shape

        cap = cv2.VideoCapture(self.source)

        ret, frame0 = cap.read()
        
        if not ret:
            return
            
        # if self.save:
        #     writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame0.shape[1], frame0.shape[0]))
        
        points = self.getPoints(frame0, self.num_points)

        self.initTrackers(frame0, points, self.num_points)

        while True:
            ret, frame = cap.read()

            height, width, _ = frame.shape

            if not ret:
                break

            tracked_points = []
            for tracker in self.trackers:
                ret, bbox = tracker.update(frame)

                if ret:
                    center = self.getTrackerCenter(bbox)
                    self.drawCircle(frame, [center[0], center[1]])
                    tracked_points.append(center)
            
            trackedT_norm = np.linalg.inv(np.array([[width+height, 0, width/2],
                            [0, width+height, height/2],
                            [0, 0, 1]]))
            coverT_norm = np.linalg.inv(np.array([[cwidth+cheight, 0, cwidth/2],
                            [0, cwidth+cheight, cheight/2],
                            [0, 0, 1]]))
            
            if len(tracked_points) == 4:
                tracked_points = np.array(tracked_points)
                cover_points_norml = (coverT_norm @ points_cover.T).T
                tracked_points_norml = (trackedT_norm @ tracked_points.T).T
                points_1 = cover_points_norml
                points_2 = tracked_points_norml

                self.A = []
                for i in range(self.num_points):
                    top = np.hstack((np.zeros(3), -1 * points_2[i][2] * points_1[i], points_2[i][1] * points_1[i]))
                    bottom = np.hstack((points_2[i][2] * points_1[i], np.zeros(3), -1 * points_2[i][0] * points_1[i]))
                    A_i = np.stack((top, bottom))
                    self.A.append(A_i)
                
                self.A = np.vstack(self.A)
                U, S, Vh = np.linalg.svd(self.A)
                h = np.reshape(Vh.T[:, -1], (3, 3))

                h = (np.linalg.inv(trackedT_norm) @ h) @ coverT_norm

                warped_img1 = cv2.warpPerspective(cover, h, (width, height))

                x = np.linspace(np.int32(tracked_points[0][0]), np.int32(tracked_points[1][0]), np.int32(tracked_points[1][0])-np.int32(tracked_points[0][0])+1)
                y = np.linspace(np.int32(tracked_points[0][1]), np.int32(tracked_points[2][1]), np.int32(tracked_points[2][1])-np.int32(tracked_points[0][1])+1)

                Xq, Yq = np.meshgrid(x, y)
                
                template = cv2.remap(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY), Xq.astype(np.float32), Yq.astype(np.float32), cv2.INTER_LINEAR)
                
                # print(np.array([(p[0], p[1]) for p in tracked_points], dtype=np.int32))
                cv2.fillConvexPoly(frame, np.array([(tracked_points[0][0], tracked_points[0][1]), 
                                                    (tracked_points[2][0], tracked_points[2][1]), 
                                                    (tracked_points[3][0], tracked_points[3][1]), 
                                                    (tracked_points[1][0], tracked_points[1][1])], dtype=np.int32), 0, 16)
                
                # frame[track] = frame + cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                frame = cv2.addWeighted(frame, 0.7, warped_img1, 0.3, 0.0)
                # frame[] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                cv2.imshow("template", template)

            cv2.imshow("AR", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    o = AugmentedReality(0)
    o.start()


