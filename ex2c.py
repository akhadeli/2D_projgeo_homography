import numpy as np
import matplotlib.pyplot as plt
import cv2

PARALLEL_MODE = 1
LTL_MODE = 2

class LineTracker:
    def __init__(self, mode, capture_source=0, patch_size=85, save=False, output_filename='output.avi'):
        self.mode = mode
        self.source = capture_source
        self.patch_size = patch_size
        self.save = save
        self.output_filename = output_filename
        self.trackers = []

    def initTrackers(self, frame, points, num_trackers):
        # x y 1
        for i in range(num_trackers):
            bbox = (np.int32(points[i][0] - (self.patch_size-1)/2), np.int32(points[i][1] - (self.patch_size-1)/2), np.int32(self.patch_size), np.int32(self.patch_size))
            self.trackers.append(cv2.TrackerKCF_create())
            self.trackers[i].init(frame, bbox)
    
    def getPoints(self, frame, num_points):
        # return cv2.selectROI("ROI", frame, fromCenter=False)
        fig, ax = plt.subplots()
        plt.imshow(frame)
        points = np.concatenate((np.array(plt.ginput(num_points)), np.ones((num_points, 1))), axis=1)
        plt.close()
        return points
    
    def getTrackerCenter(self, bbox):
        return np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, 1], dtype=np.int32)
    
    def getLines(self, p):
        # Assuming (line: indexes)
        # l1: 0 1
        # l2: 2 3
        # l3: 4 5
        # l4: 6 7
        return np.cross(p[0], p[1]), np.cross(p[2], p[3]), np.cross(p[4], p[5]), np.cross(p[6], p[7])
    
    def drawLine(self, frame, p1, p2, color=(255, 0, 0), thickness=2):
        cv2.line(frame, p1, p2, color, thickness)
    
    def drawCircle(self, frame, center, radius=2, color=(0, 0, 255), thickness=1):
        cv2.circle(frame, center, radius, color, thickness)
    
    def drawText(self, frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1):
        cv2.putText(frame, text, org, font, fontScale, color, thickness)
    
    def errorParallel(self, l1, l2, l3, l4):
        return np.cross(np.cross(l1, l2), np.cross(l3, l4))
    
    def errorLineToLine(self, y1, y2, y3, y4):
        y3y4 = np.cross(y3, y4)
        norm_y3y4 = y3y4/y3y4[2]
        return np.dot(y1, norm_y3y4) + np.dot(y2, norm_y3y4)
    
    def start(self):
        if self.mode == PARALLEL_MODE:
            self.start_parallel()
        elif self.mode == LTL_MODE:
            self.start_ltl()
        else:
            return
    
    def start_parallel(self):
        cap = cv2.VideoCapture(self.source)

        ret, frame0 = cap.read()
        if not ret:
            print("failed to capture frame")
            return
        
        if self.save:
            writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame0.shape[1], frame0.shape[0]))
        
        points = self.getPoints(frame0, 8)
        fixed_points = np.array(points[4:8],dtype=np.int32)

        self.initTrackers(frame0, points, 4)

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            tracked_points = []
            for tracker in self.trackers:
                ret, bbox = tracker.update(frame)

                if ret:
                    center = self.getTrackerCenter(bbox)
                    self.drawCircle(frame, [center[0], center[1]])
                    tracked_points.append(center)
            
            tracked_points = np.concatenate((tracked_points, fixed_points), axis=0)

            if len(tracked_points) == 8:
                l1, l2, l3, l4 = self.getLines(tracked_points)

                epar = self.errorParallel(l1, l2, l3, l4)
                self.drawText(frame, 'epar:', (10,10), fontScale=0.5)
                self.drawText(frame, f'norm:{str(np.linalg.norm(epar[:2]/epar[2]))}', (10,40), fontScale=0.3)

                for i in range(1, len(tracked_points), 2):
                    self.drawLine(frame, (tracked_points[i-1][0], tracked_points[i-1][1]), (tracked_points[i][0], tracked_points[i][1]))

            cv2.imshow("Parallel lines", frame)

            if self.save:
                writer.write(frame)

            key = cv2.waitKey(100)
            if key == 27:
                break
        
        cv2.destroyAllWindows()

    def start_ltl(self):
        cap = cv2.VideoCapture(self.source)

        ret, frame0 = cap.read()
        if not ret:
            print("failed to capture frame")
            return
            
        if self.save:
            writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame0.shape[1], frame0.shape[0]))

        points = self.getPoints(frame0, 4)
        fixed_points = np.array(points[2:4],dtype=np.int32)

        self.initTrackers(frame0, points, 2)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            tracked_points = []
            for tracker in self.trackers:
                ret, bbox = tracker.update(frame)

                if ret:
                    center = self.getTrackerCenter(bbox)
                    self.drawCircle(frame, [center[0], center[1]])
                    tracked_points.append(center)
            
            tracked_points = np.concatenate((tracked_points, fixed_points), axis=0)
            
            if len(tracked_points) == 4:
                ell = self.errorLineToLine(*tracked_points)
                self.drawText(frame, 'ell:', (10,10), fontScale=0.5)
                self.drawText(frame, f'norm:{str(ell)}', (10,40), fontScale=0.3)
                for i in range(1, len(tracked_points), 2):
                    self.drawLine(frame, (tracked_points[i-1][0], tracked_points[i-1][1]), (tracked_points[i][0], tracked_points[i][1]))

            cv2.imshow("line to line", frame)

            if self.save:
                writer.write(frame)

            key = cv2.waitKey(30)
            if key == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    o = LineTracker(LTL_MODE, '2c_ltl.avi')
    o.start()