import time
import json
from params import *
from stabilization import *
from background_subtraction import *
from matting import *
from tracking import *

def main():

    timing_log = {}

    # Video Stabilization
    start_time = time.time()
    video_Stabilization(INPUT_PATH, STAB_PATH)  
    end_time = time.time()
    timing_log["time_to_stabilize"] = end_time - start_time

    # Background Subtraction
    start_time = time.time()
    background_subtraction(STAB_PATH, BINARY_PATH, EXTRACTED_PATH)
    end_time = time.time()
    timing_log["time_to_binary"] = end_time - start_time

    # Matting
    start_time = time.time()
    matting(STAB_PATH, BINARY_PATH, NEW_BG_PATH, ALPHA_PATH, MATTED_PATH)
    end_time = time.time()
    timing_log["time_to_matted"] = end_time - start_time

    # Tracking
    start_time = time.time()
    tracking(MATTED_PATH, ALPHA_PATH, OUTPUT_PATH, TRACKING_LOG_PATH)
    end_time = time.time()
    timing_log["time_to_output"] = end_time - start_time

    with open(TIMING_LOG_PATH, 'w') as fp:
        json.dump(timing_log, fp, indent=4)


if __name__ == "__main__":
    main()