import numpy as np
import cv2
from params import *

""" General """

def video_to_frame_list(cap, resize=None):
  frames_list = []
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    if resize is not None:
      frame = cv2.resize(frame, resize)
    frames_list.append(frame)
  return np.array(frames_list)


""" Video Stabilization"""

def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  kernel = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, kernel, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory, smoothing_radius):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(trajectory.shape[1]):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=smoothing_radius)
  return smoothed_trajectory

def fixBorder(frame):
  h, w = frame.shape[:2]
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((w/2, h/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (w, h))
  return frame


""" Background Subtraction """

def kde_pdf_memo(memo_dict, pixval, pdf):
  if tuple(pixval) not in memo_dict:
    memo_dict[tuple(pixval)] = pdf(pixval.T)[0]
  return memo_dict[tuple(pixval)]

def contours_filtering(mask):
  """ Find contours and fill them with black (except the largest).
  Returns the filtered mask and the largest contour.  """
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)
  cv2.drawContours(mask, contours[1:], -1, 0, thickness=cv2.FILLED)
  return mask, contours[0]