import os

""" IDs """
ID1 = 315803148
ID2 = 307875161

""" Paths """
INPUT_DIR = "Inputs"
OUTPUT_DIR = "Outputs"
TEMP_DIR = "Temp"
INPUT_NAME = "INPUT.avi"
INPUT_PATH = os.path.join(INPUT_DIR, INPUT_NAME)
TIMING_LOG_NAME = "timing.json"
TIMING_LOG_PATH = os.path.join(OUTPUT_DIR, TIMING_LOG_NAME)
TRACKING_LOG_NAME = "tracking.json"
TRACKING_LOG_PATH = os.path.join(OUTPUT_DIR, TRACKING_LOG_NAME)
STAB_NAME = f"stabilized_{ID1}_{ID2}.avi"
STAB_PATH = os.path.join(OUTPUT_DIR, STAB_NAME)
BINARY_NAME = f"binary_{ID1}_{ID2}.avi"
BINARY_PATH = os.path.join(OUTPUT_DIR, BINARY_NAME)
EXTRACTED_NAME = f"extracted_{ID1}_{ID2}.avi"
EXTRACTED_PATH = os.path.join(OUTPUT_DIR, EXTRACTED_NAME)
ALPHA_NAME = f"alpha_{ID1}_{ID2}.avi"
ALPHA_PATH = os.path.join(OUTPUT_DIR, ALPHA_NAME)
MATTED_NAME = f"matted_{ID1}_{ID2}.avi"
MATTED_PATH = os.path.join(OUTPUT_DIR, MATTED_NAME)
NEW_BG_NAME = "background.jpg"
NEW_BG_PATH = os.path.join(INPUT_DIR, NEW_BG_NAME)
OUTPUT_NAME = f"OUTPUT_{ID1}_{ID2}.avi"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

""" Video Stabilization """
# Parameters for goodFeaturesToTrack
MAX_CORNERS = 1000
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10
BLOCK_SIZE = 5
# Parameters for calcOpticalFlowPyrLK
MAX_LVL = 5
WIN_SIZE = 21
# Parameters for findHomography
RANSAC_THRESH = 5.0
# Parameters for smooth
SMOOTHING_RADIUS = 5  

""" Background Subtraction """
BGSUB_ITER = 5
KNN_DIST2THRESH = 400
SE_KSIZE = 15
BINARY_THRESH = 200
MEDIAN_KSIZE = 7
BS_FG_SAMPLES = 20
BS_BG_SAMPLES = 60 
BS_ENLARGEMENT = 0.2

""" Matting """
M_FG_SAMPLES = 100
M_BG_SAMPLES = 100 
ED_KSIZE = 5
GEODIST_ITER = 1
DELTA = 0.03
R = 2
M_ENLARGEMENT = 0.2

""" Tracking """
MASK_LOW_H = 0.
MASK_LOW_S = 70.
MASK_LOW_V = 40.
MASK_HIGH_H = 180.
MASK_HIGH_S = 255.
MASK_HIGH_V = 255.