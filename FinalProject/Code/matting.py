import cv2
import numpy as np
import GeodisTK
from scipy.stats import gaussian_kde
from tqdm import tqdm
from params import *
from utils import *

""" https://github.com/taigw/GeodisTK """

def matting(stab_path, binary_path, backgroud_path, alpha_path, matted_path):

    cap = cv2.VideoCapture(stab_path)
    cap_bin = cv2.VideoCapture(binary_path)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size =  (width, height)

    alpha_out = cv2.VideoWriter(alpha_path, fourcc, fps, frame_size, isColor=False)
    matted_out = cv2.VideoWriter(matted_path, fourcc, fps, frame_size)

    print("[Matting] started.")

    new_backgroud = cv2.imread(backgroud_path)
    new_backgroud = cv2.resize(new_backgroud, frame_size)

    for _ in tqdm(range(frame_count), desc="Matting frames"):

        ret, frame = cap.read()
        if not ret:
            break

        ret, binary_frame = cap_bin.read()
        if not ret:
            break
        
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        luma = cv2.split(frame_yuv)[0]
        
        binary_white_indices = np.where(binary_frame == 255)
        x_start, x_end = np.min(binary_white_indices[1]), np.max(binary_white_indices[1])
        y_start, y_end = np.min(binary_white_indices[0]), np.max(binary_white_indices[0])
        w_enlarge = int((x_end - x_start) * M_ENLARGEMENT)
        h_enlarge = int((y_end - y_start) * M_ENLARGEMENT)
        x_start, x_end = max(x_start - w_enlarge, 0), min(x_end + w_enlarge, width)
        y_start, y_end = max(y_start - h_enlarge, 0), min(y_end + h_enlarge ,height)

        frame_cropped = frame[y_start:y_end, x_start:x_end]
        bin_frame_cropped  = binary_frame[y_start:y_end, x_start:x_end]
        luma_cropped  = luma[y_start:y_end, x_start:x_end]
        new_backgroud_cropped  = new_backgroud[y_start:y_end, x_start:x_end]

        fg_scribble = cv2.erode(bin_frame_cropped , np.ones((ED_KSIZE,ED_KSIZE)))
        fg_dist_map = GeodisTK.geodesic2d_raster_scan(luma_cropped , fg_scribble, 1.0, GEODIST_ITER)

        bg_scribble = cv2.dilate(bin_frame_cropped , np.ones((ED_KSIZE,ED_KSIZE)))
        bg_scribble = (255 - bg_scribble).astype(np.uint8)
        bg_dist_map = GeodisTK.geodesic2d_raster_scan(luma_cropped , bg_scribble, 1.0, GEODIST_ITER)

        margin = (fg_dist_map + bg_dist_map)*DELTA
        undecided_indices = np.where(np.abs(fg_dist_map - bg_dist_map) < margin)
        fg_decided = np.where(fg_dist_map + margin < bg_dist_map , 1, 0)
        bg_decided = np.where(bg_dist_map + margin < fg_dist_map, 1, 0)

        # Sample random pixels from Foreground
        fg = frame_cropped[fg_decided==1]
        fg_random_indices = np.random.choice(fg.shape[0], size=M_FG_SAMPLES, replace=False)
        fg_sampled_pixels = fg[fg_random_indices]

        # Sample random pixels from Background
        bg = frame_cropped[bg_decided==1]
        bg_random_indices = np.random.choice(bg.shape[0], size=M_BG_SAMPLES, replace=False)
        bg_sampled_pixels = bg[bg_random_indices]

        # Estimate Background & Foreground PDFs
        fg_pdf = gaussian_kde(fg_sampled_pixels.T)
        bg_pdf = gaussian_kde(bg_sampled_pixels.T)

        # Calculate Background & Foreground Probabilities
        P_fg = fg_pdf(frame_cropped[undecided_indices].T)
        P_bg = bg_pdf(frame_cropped[undecided_indices].T)

        # Trimap refinement
        W_fg = np.power(fg_dist_map[undecided_indices], -R) * P_fg
        W_bg = np.power(bg_dist_map[undecided_indices], -R) * P_bg

        undecided_alpha = W_fg / (W_fg + W_bg)
        alpha_cropped = fg_decided.copy().astype(np.float32)
        alpha_cropped[undecided_indices] = undecided_alpha

        # Matting
        matted_cropped = alpha_cropped[:, :, np.newaxis] * frame_cropped  + (1 - alpha_cropped[:, :, np.newaxis]) * new_backgroud_cropped 

        alpha = np.zeros((height, width), dtype=np.uint8)
        alpha[y_start:y_end, x_start:x_end] = (255* alpha_cropped).astype(np.uint8)
        
        matted = new_backgroud.copy()
        matted[y_start:y_end, x_start:x_end] = matted_cropped 

        alpha_out.write(alpha)
        matted_out.write(matted)

    print("[Matting] finished.")

    cap.release()
    alpha_out.release()
    matted_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    matting(STAB_PATH, BINARY_PATH, NEW_BG_PATH, ALPHA_PATH, MATTED_PATH)
