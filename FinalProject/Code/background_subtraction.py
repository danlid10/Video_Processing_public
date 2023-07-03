import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import gaussian_kde
from params import *
from utils import *

def background_subtraction(stab_path, binary_path, extracted_path):
    
    cap = cv2.VideoCapture(stab_path)
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size =  (width, height)

    binary_out = cv2.VideoWriter(binary_path, fourcc, fps, frame_size, isColor=False)
    extracted_out = cv2.VideoWriter(extracted_path, fourcc, fps, frame_size)

    print("[Background Subtraction] started.")

    fg_masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    frames_contours = []
    
    bgsub = cv2.createBackgroundSubtractorKNN(dist2Threshold=KNN_DIST2THRESH)

    for _ in tqdm(range(BGSUB_ITER), desc="BackgroundSubtractor iterations"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            fg_masks[i] = bgsub.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SE_KSIZE, SE_KSIZE))

    fg_sampled_pixels, bg_sampled_pixels  = None, None
    fg_memo_dict, bg_memo_dict = {}, {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    for mask in tqdm(fg_masks, desc="Collecting samples for KDEs"):

        ret, frame = cap.read()
        if not ret:
            break

        mask = cv2.medianBlur(mask, MEDIAN_KSIZE)
        mask = cv2.threshold(mask, BINARY_THRESH, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask, contour = contours_filtering(mask)
        frames_contours.append(contour)

        fg = frame[mask==255]
        random_indices = np.random.choice(fg.shape[0], size=BS_FG_SAMPLES, replace=False)
        sampled_pixels = fg[random_indices]
        if fg_sampled_pixels is None:
            fg_sampled_pixels = np.array(sampled_pixels)
        else:
            fg_sampled_pixels = np.concatenate((fg_sampled_pixels, sampled_pixels))

        bg = frame[mask==0]
        random_indices = np.random.choice(bg.shape[0], size=BS_BG_SAMPLES, replace=False)
        sampled_pixels = bg[random_indices]
        if bg_sampled_pixels is None:
            bg_sampled_pixels = np.array(sampled_pixels)
        else:
            bg_sampled_pixels = np.concatenate((bg_sampled_pixels, sampled_pixels))

    fg_kde = gaussian_kde(fg_sampled_pixels.T)
    bg_kde = gaussian_kde(bg_sampled_pixels.T)
    calc_fg_density = lambda pixel : kde_pdf_memo(fg_memo_dict, pixel, fg_kde)
    calc_bg_density = lambda pixel : kde_pdf_memo(bg_memo_dict, pixel, bg_kde)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    for i in tqdm(range(frame_count), desc="Applying KDEs"):

        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = cv2.boundingRect(frames_contours[i])
        w_enlargement, h_enlargement = int(w * BS_ENLARGEMENT), int(h * BS_ENLARGEMENT)
        x, y = x - w_enlargement, y - h_enlargement
        w, h = w + 2 * w_enlargement,  h + 2 * h_enlargement
        x_start, x_end = max(x, 0), min(x + w, width)
        y_start, y_end = max(y, 0), min(y + h, height)
        rect_w, rect_h = x_end - x_start, y_end - y_start

        pixels = frame[y_start:y_end, x_start:x_end].reshape(-1, 3)

        fg_density = np.fromiter(map(calc_fg_density, pixels), np.float32)
        bg_density = np.fromiter(map(calc_bg_density, pixels), np.float32)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y_start:y_end, x_start:x_end] = np.where(fg_density > bg_density, 255, 0).reshape((rect_h, rect_w)).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask, _ = contours_filtering(mask)

        extracted = cv2.bitwise_and(frame, frame, mask=mask)

        binary_out.write(mask)
        extracted_out.write(extracted)

    print("[Background Subtraction] finished.")

    cap.release()
    binary_out.release()
    extracted_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    background_subtraction(STAB_PATH, BINARY_PATH, EXTRACTED_PATH)
