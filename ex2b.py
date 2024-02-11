import numpy as np
import matplotlib.pyplot as plt
import cv2

# Points selected in following order:
# 0: bottom-left
# 1: top-right
# 2: bottom-right
# 3: top-left

class ObjectMidpoint:
    def __init__(self, capture_source=0, patch_size=45, save=False, output_filename='output.avi'):
        self.source = capture_source
        self.patch_size = patch_size
        self.trackers = []
        self.save = save
        self.output_filename = output_filename

    def getPoints(self, frame):
        # return cv2.selectROI("ROI", frame, fromCenter=False)
        fig, ax = plt.subplots()
        plt.imshow(frame)
        return np.concatenate((np.array(plt.ginput(4)), np.ones((4, 1))), axis=1)
    
    def initTrackers(self, frame, points):
        # x y 1
        for i, point in enumerate(points):
            bbox = (np.int32(point[0] - (self.patch_size-1)/2), np.int32(point[1] - (self.patch_size-1)/2), np.int32(self.patch_size), np.int32(self.patch_size))
            self.trackers.append(cv2.TrackerKCF_create())
            self.trackers[i].init(frame, bbox)


    def calcMidpoint(self, points):
        l1 = np.cross(points[0], points[1])
        l2 = np.cross(points[2], points[3])

        x_m = np.cross(l1, l2)

        return np.array([x_m[0]/x_m[2], x_m[1]/x_m[2]], dtype=np.int32)
    
    def calcVanish(self, points):
        l3 = np.cross(points[0], points[2])
        l4 = np.cross(points[1], points[3])

        x_inf = np.cross(l3, l4)

        return np.array([x_inf[0]/x_inf[2], x_inf[1]/x_inf[2]], dtype=np.int32)
    
    def start(self):
        cap = cv2.VideoCapture(self.source)


        ret, frame0 = cap.read()
        frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        
        if not ret:
            return
            
        if self.save:
            writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame0.shape[1], frame0.shape[0]))
        
        points = self.getPoints(frame0)

        self.initTrackers(frame0, points)

        while True:
            ret, frame = cap.read()
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break
            
            tracked_points = []
            for tracker in self.trackers:
                ret, bbox = tracker.update(frame)

                if ret:
                    # cv2.rectangle(frame, (np.int32(bbox[0]), np.int32(bbox[1])), (np.int32(bbox[0]+bbox[2]), np.int32(bbox[1]+bbox[3])), (0, 255, 0), 1)
                    center = [np.int32(bbox[0] + bbox[2]/2), np.int32(bbox[1] + bbox[3]/2), 1]
                    cv2.circle(frame, (center[0], center[1]), 2, (0, 0, 255), 2)
                    tracked_points.append(center)
                    
            if len(tracked_points) == 4:
                midpoint = self.calcMidpoint(tracked_points)
                cv2.circle(frame, midpoint, 2, (0, 0, 255), 2)
                cv2.line(frame, (tracked_points[0][0], tracked_points[0][1]), (tracked_points[1][0], tracked_points[1][1]), (255, 0, 0), 1)
                cv2.line(frame, (tracked_points[2][0], tracked_points[2][1]), (tracked_points[3][0], tracked_points[3][1]), (255, 0, 0), 1)

                try:
                    vanish_point = self.calcVanish(tracked_points)
                    cv2.line(frame, vanish_point, midpoint, (0, 255, 0), 1)
                except:
                    pass

            if self.save:
                writer.write(frame)

            cv2.imshow("Tracker", frame)
            key = cv2.waitKey(30)
            if key == 27:
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    o = ObjectMidpoint('2b.mp4', save=True)
    o.start()