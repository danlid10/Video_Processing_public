import numpy as np
import cv2
from tqdm import tqdm
import json
from params import *

""" https://docs.opencv.org/3.4/d7/d00/tutorial_meanshift.html """

def tracking(matted_path, alpha_path, output_path, tracking_log_path):

    cap = cv2.VideoCapture(matted_path)
    cap_alpha = cv2.VideoCapture(alpha_path)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (width, height)

    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    print("[Tracking] started.")

    tracking_log = {}

    ret, alpha_frame = cap_alpha.read()

    # Get initial track window form the first alpha frame
    alpha_white_indices = np.where(alpha_frame == 255)
    x, y = np.min(alpha_white_indices[1]), np.min(alpha_white_indices[0])
    w, h = np.max(alpha_white_indices[1]) - x, np.max(alpha_white_indices[0]) - y
    track_window = (x, y, w, h)
    tracking_log[1] = [int(i) for i in [x, y, h, w]]

    ret, frame = cap.read()

    tracked = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
    out.write(tracked)

    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((MASK_LOW_H, MASK_LOW_S, MASK_LOW_V)),  np.array((MASK_HIGH_H, MASK_HIGH_S, MASK_HIGH_V)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    for i in tqdm(range(1, frame_count), desc="Tracking frames"):

        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x ,y ,w ,h = track_window
        tracking_log[i+1] = [int(i) for i in [x, y, h, w]]

        tracked = cv2.rectangle(frame, (x,y), (x+w,y+h), color=(0, 255, 0), thickness=2)

        out.write(tracked)

    with open(tracking_log_path, 'w') as fp:
        json.dump(tracking_log, fp, indent=4)

    print("[Tracking] finished.")

    cap.release()
    cap_alpha.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tracking(MATTED_PATH, ALPHA_PATH, OUTPUT_PATH, TRACKING_LOG_PATH)