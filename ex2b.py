import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
import cv2

class objectMidpoint:
    def __init__(self, capture_source=0):
        self.source = capture_source
        self.tracker = cv2.TrackerKCF_create()

    def getPoints(self, frame):
        return cv2.selectROI("ROI", frame, fromCenter=False)
    
    def calcMidpoint(self, bbox):
        top_left = np.array([bbox[0], bbox[1], 1])
        top_right = np.array([bbox[0]+bbox[2], bbox[1], 1])
        bottom_left = np.array([bbox[0], bbox[1]+bbox[3], 1])
        bottom_right = np.array([bbox[0]+bbox[2], bbox[1]+bbox[3], 1])

        l1 = np.cross(top_left, bottom_right)
        l2 = np.cross(top_right, bottom_left)

        x_m = np.cross(l1, l2)

        return np.array([x_m[0]/x_m[2], x_m[1]/x_m[2]], dtype=np.int32)
    
    def calcVanish(self, bbox):
        top_left = np.array([bbox[0], bbox[1], 1])
        top_right = np.array([bbox[0]+bbox[2], bbox[1], 1])
        bottom_left = np.array([bbox[0], bbox[1]+bbox[3], 1])
        bottom_right = np.array([bbox[0]+bbox[2], bbox[1]+bbox[3], 1])

        l3 = np.cross(top_left, top_right)
        l4 = np.cross(bottom_right, bottom_left)

        x_inf = np.cross(l3, l4)

        return np.array([x_inf[0]/x_inf[2], x_inf[1]/x_inf[2]], dtype=np.int32)
    
    def start(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        ret, frame0 = cap.read()
        if not ret:
            return
        
        bbox = self.getPoints(frame0)
        print(bbox)
        ret = self.tracker.init(frame0, bbox)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ret, bbox = self.tracker.update(frame)

            if ret:
                cv2.rectangle(frame, (np.int32(bbox[0]), np.int32(bbox[1])), (np.int32(bbox[0]+bbox[2]), np.int32(bbox[1]+bbox[3])), (0, 255, 0), 1)
                
                cv2.line(frame, np.array([bbox[0], bbox[1]]), np.array([bbox[0]+bbox[2], bbox[1]+bbox[3]]), (255, 0, 0), 1)
                cv2.line(frame, np.array([bbox[0]+bbox[2], bbox[1]]), np.array([bbox[0], bbox[1]+bbox[3]]), (255, 0, 0), 1)
                
                midpoint = self.calcMidpoint(bbox)
                cv2.circle(frame, midpoint, 2, (0, 0, 255), 2)
                
                try:
                    vanish_point = self.calcVanish(bbox)
                    cv2.line(frame, vanish_point, midpoint, (0, 255, 0), 1)
                except:
                    pass

            cv2.imshow("Tracker", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    o = objectMidpoint('test.mp4')
    o.start()