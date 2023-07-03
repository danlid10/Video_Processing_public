import numpy as np
import cv2
from tqdm import tqdm
from params import *
from utils import *

""" https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/ """

def video_Stabilization(video_path, out_path):

  cap = cv2.VideoCapture(video_path)

  fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_size =  (width, height)
  
  out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)

  print("[Video Stabilization] started.")

  feature_params = dict(maxCorners=MAX_CORNERS, qualityLevel=QUALITY_LEVEL, minDistance=MIN_DISTANCE, blockSize=BLOCK_SIZE)
  LK_params = dict(maxLevel=MAX_LVL, winSize=(WIN_SIZE, WIN_SIZE))
  homography_params = dict(method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH)

  transforms = []

  prev_gray = None

  for _ in tqdm(range(frame_count), desc="Collecting Transformations"):
          
    ret, frame = cap.read()
    if not ret:
      break

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    if prev_gray is None:
      prev_gray = curr_gray
      continue

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **LK_params) 
    
    prev_pts = prev_pts[status==1]
    curr_pts = curr_pts[status==1]

    H, _ = cv2.findHomography(prev_pts, curr_pts, **homography_params) 

    transforms.append(H.flatten())

    prev_gray = curr_gray

  trajectory = np.cumsum(transforms, axis=0)

  smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS)

  difference = smoothed_trajectory - trajectory
  
  transforms_smooth = transforms + difference

  # Write first frame to output
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
  ret, frame = cap.read()
  out.write(fixBorder(frame))

  cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
  for H_flat in tqdm(transforms_smooth, desc="Stabilizing frames"):

    ret, frame = cap.read()
    if not ret:
      break

    H = H_flat.reshape((3,3))

    frame_stabilized = cv2.warpPerspective(frame, H, frame_size)
  
    frame_stabilized = fixBorder(frame_stabilized) 
  
    out.write(frame_stabilized)

  print("[Video Stabilization] finished.")

  cap.release()
  out.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  video_Stabilization(INPUT_PATH, STAB_PATH)  
