import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
from numpy.lib.stride_tricks import sliding_window_view     # Added

# FILL IN YOUR ID
ID1 = 42
ID2 = 42

PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5

""" Added Constants - used in: faster_lucas_kanade_step"""
K = 0.05            # Parameter for Harris corner detector
SMALL_FACTOR = 10   # The image is small enough if: image dimensions < SMALL_FACTOR * window_size

""" Added debugging for video stabilization """
DEBUG = False
if DEBUG:
    import os
    DEBUG_DIR = 'DEBUG'
    NAIVE_DIR = os.path.join(DEBUG_DIR, 'NAIVE')
    FASTER_DIR = os.path.join(DEBUG_DIR, 'FASTER')
    FASTER_FIX_DIR = os.path.join(DEBUG_DIR, 'FASTER_FIX')
    os.makedirs(NAIVE_DIR, exist_ok=True)
    os.makedirs(FASTER_DIR, exist_ok=True)
    os.makedirs(FASTER_FIX_DIR, exist_ok=True)


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width, "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    """INSERT YOUR CODE HERE."""
    for level in range(num_levels):
        blurred = signal.convolve2d(pyramid[level], PYRAMID_FILTER, 'same', 'symm')
        decimated = blurred[::2, ::2]
        pyramid.append(decimated)

    return pyramid


def lucas_kanade_step(I1: np.ndarray, I2: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, 'same', 'symm')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, 'same', 'symm')
    It = I2.astype(np.int32) - I1.astype(np.int32)
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    Ix_windows = sliding_window_view(Ix, (window_size, window_size))
    Iy_windows = sliding_window_view(Iy, (window_size, window_size))
    It_windows = sliding_window_view(It, (window_size, window_size))

    boundary = window_size//2
    for i in range(Ix_windows.shape[0]):
        for j in range(Ix_windows.shape[1]):
            Ix_window = Ix_windows[i, j].reshape(-1, 1)
            Iy_window = Iy_windows[i, j].reshape(-1, 1)
            It_window = It_windows[i, j].reshape(-1, 1)
            B = np.hstack((Ix_window, Iy_window))
            BTB = B.T @ B
            if np.linalg.det(BTB) != 0:  # Invertable
                Delta_P = - np.linalg.inv(BTB) @ B.T @ It_window
                du[i + boundary, j + boundary] = Delta_P[0, 0]
                dv[i + boundary, j + boundary] = Delta_P[1, 0]

    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    """INSERT YOUR CODE HERE.
    Replace image_warp with something else.
    """
    h, w = image.shape[:2]
    if image.shape != u.shape:
        u_factor = image.shape[1] / u.shape[1]
        u = cv2.resize(u, (w, h)) * u_factor
    if image.shape != v.shape:
        v_factor = image.shape[0] / v.shape[0]
        v = cv2.resize(v, (w, h)) * v_factor

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.column_stack((x.ravel(), y.ravel()))
    values = image.ravel()
    interpolate_points = np.column_stack(((x+u).ravel(), (y+v).ravel()))
    image_warp = griddata(points, values, interpolate_points).reshape((h, w))
    image_warp[np.isnan(image_warp)] = image[np.isnan(image_warp)]

    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^num_levels) X M * num_levels).
        Where: K is the ceil(h / (2^num_levels), and M is ceil(w / (2^num_levels).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.T.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.T.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""

    for level in range(num_levels, -1, -1):
        I2_warp = warp_image(pyarmid_I2[level], u, v)
        for i in range(max_iter):
            du, dv = lucas_kanade_step(pyramid_I1[level], I2_warp, window_size)
            u, v = u + du, v + dv
            I2_warp = warp_image(pyarmid_I2[level], u, v)

        if level > 0:
            u = 2 * cv2.resize(u, tuple([2*n for n in u.T.shape]))
            v = 2 * cv2.resize(v, tuple([2*n for n in v.T.shape]))

    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^num_levels) X M * (2^num_levels).
        Where: K is the ceil(h / (2^num_levels), and M is ceil(w / (2^num_levels).
        (5) Create a u and a v which are of the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    boundry = window_size//2
    cap = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = parameters['fps']
    framesize = (parameters['width'], parameters['height'])
    frame_count = parameters['frame_count']

    # Calculate the desired image size - in opencv shape format (w,h)
    h_factor = int(np.ceil(framesize[1] / (2 ** num_levels)))
    w_factor = int(np.ceil(framesize[0] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))

    prev_u, prev_v = np.zeros(IMAGE_SIZE[::-1]), np.zeros(IMAGE_SIZE[::-1])

    out = cv2.VideoWriter(output_video_path, fourcc, fps, IMAGE_SIZE, isColor=False)
    print('Lucas-Kanade video stabilization:')
    pbar = tqdm(total=frame_count)

    ret, prev_frame = cap.read()
    if ret:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.resize(prev_frame, IMAGE_SIZE)
        
        if DEBUG:
            path = os.path.join(NAIVE_DIR, 'frame_1.png')
            cv2.imwrite(path, prev_frame)
            frame_num = 1

        out.write(np.uint8(prev_frame))
        pbar.update(1)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, IMAGE_SIZE)
            u, v = lucas_kanade_optical_flow(prev_frame, frame, window_size, max_iter, num_levels)
            u[boundry:-boundry, boundry:-boundry] = np.mean(u[boundry:-boundry, boundry:-boundry])
            v[boundry:-boundry, boundry:-boundry] = np.mean(v[boundry:-boundry, boundry:-boundry])
            out_frame = warp_image(frame, prev_u + u, prev_v + v)

            if DEBUG:
                frame_num += 1
                path = os.path.join(NAIVE_DIR, f'frame_{frame_num}.png')
                cv2.imwrite(path, out_frame)

            out.write(np.uint8(out_frame))
            pbar.update(1)

            prev_u, prev_v = prev_u + u, prev_v + v
            prev_frame = frame

        else:
            break

    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    if I1.shape[0] < SMALL_FACTOR*window_size or I1.shape[1] < SMALL_FACTOR*window_size:
        return lucas_kanade_step(I1, I2, window_size)

    boundary = window_size//2

    # Find corners in I2 using Harris corner detector
    I2_corners = cv2.cornerHarris(np.uint8(I2), window_size, 3, K)
    I2_corners = cv2.dilate(I2_corners, None)
    threshold = 0.01 * I2_corners.max()
    I2_corners_no_edges = np.zeros_like(I2)
    I2_corners_no_edges[boundary:-boundary, boundary:-boundary] = I2_corners[boundary:-boundary, boundary:-boundary]
    I2_corners_args = np.argwhere(I2_corners_no_edges > threshold)

    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, 'same', 'symm')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, 'same', 'symm')
    It = I2.astype(np.int32) - I1.astype(np.int32)
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    Ix_windows = sliding_window_view(Ix, (window_size, window_size))
    Iy_windows = sliding_window_view(Iy, (window_size, window_size))
    It_windows = sliding_window_view(It, (window_size, window_size))

    for corner_arg in I2_corners_args:
        i, j = corner_arg
        Ix_window = Ix_windows[i - boundary, j - boundary].reshape(-1, 1)
        Iy_window = Iy_windows[i - boundary, j - boundary].reshape(-1, 1)
        It_window = It_windows[i - boundary, j - boundary].reshape(-1, 1)
        B = np.hstack((Ix_window, Iy_window))
        BTB = B.T @ B
        if np.linalg.det(BTB) != 0:  # Invertable
            Delta_P = - np.linalg.inv(BTB) @ B.T @ It_window
            du[i, j] = Delta_P[0, 0]
            dv[i, j] = Delta_P[1, 0]

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.T.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.T.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    # create u in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    # create v in the size of smallest image
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        I2_warp = warp_image(pyarmid_I2[level], u, v)
        for i in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[level], I2_warp, window_size)
            u, v = u + du, v + dv
            I2_warp = warp_image(pyarmid_I2[level], u, v)

        if level > 0:
            u = 2 * cv2.resize(u, tuple([2*n for n in u.T.shape]))
            v = 2 * cv2.resize(v, tuple([2*n for n in v.T.shape]))

    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    boundry = window_size//2
    cap = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = parameters['fps']
    framesize = (parameters['width'], parameters['height'])
    frame_count = parameters['frame_count']
  
    # Calculate the desired image size - in opencv shape format (w,h)
    h_factor = int(np.ceil(framesize[1] / (2 ** num_levels)))
    w_factor = int(np.ceil(framesize[0] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))

    prev_u, prev_v = np.zeros(IMAGE_SIZE[::-1]), np.zeros(IMAGE_SIZE[::-1])

    out = cv2.VideoWriter(output_video_path, fourcc, fps, IMAGE_SIZE, isColor=False)
    print('Faster Lucas-Kanade video stabilization:')
    pbar = tqdm(total=frame_count)

    ret, prev_frame = cap.read()
    if ret:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.resize(prev_frame, IMAGE_SIZE)

        if DEBUG:
            path = os.path.join(FASTER_DIR, 'frame_1.png')
            cv2.imwrite(path, prev_frame)
            frame_num = 1

        out.write(np.uint8(prev_frame))
        pbar.update(1)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, IMAGE_SIZE)
            u, v = faster_lucas_kanade_optical_flow(prev_frame, frame, window_size, max_iter, num_levels)
            u[boundry:-boundry, boundry:-boundry] = np.mean(u[boundry:-boundry, boundry:-boundry])
            v[boundry:-boundry, boundry:-boundry] = np.mean(v[boundry:-boundry, boundry:-boundry])
            out_frame = warp_image(frame, prev_u + u, prev_v + v)

            if DEBUG:
                frame_num += 1
                path = os.path.join(FASTER_DIR, f'frame_{frame_num}.png')
                cv2.imwrite(path, out_frame)

            out.write(np.uint8(out_frame))
            pbar.update(1)

            prev_u, prev_v = prev_u + u, prev_v + v
            prev_frame = frame

        else:
            break

    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    boundry = window_size//2
    cap = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = parameters['fps']
    framesize = (parameters['width'], parameters['height'])
    frame_count = parameters['frame_count']
    
    # Calculate the desired image size - in opencv shape format (w,h)
    h_factor = int(np.ceil(framesize[1] / (2 ** num_levels)))
    w_factor = int(np.ceil(framesize[0] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels), h_factor * (2 ** num_levels))

    prev_u, prev_v = np.zeros(IMAGE_SIZE[::-1]), np.zeros(IMAGE_SIZE[::-1])

    out = cv2.VideoWriter(output_video_path, fourcc, fps, IMAGE_SIZE, isColor=False)
    print('Faster Lucas-Kanade video stabilization with fixed effects:')
    pbar = tqdm(total=frame_count)

    ret, prev_frame = cap.read()
    if ret:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Fix border effects by cutting out a constant portion of the image
        prev_frame = prev_frame[start_rows:-end_rows, start_cols:-end_cols]
        prev_frame = cv2.resize(prev_frame, IMAGE_SIZE)

        if DEBUG:
            path = os.path.join(FASTER_FIX_DIR, 'frame_1.png')
            cv2.imwrite(path, prev_frame)
            frame_num = 1

        out.write(np.uint8(prev_frame))
        pbar.update(1)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, IMAGE_SIZE)
            u, v = faster_lucas_kanade_optical_flow(prev_frame, frame, window_size, max_iter, num_levels)
            u[boundry:-boundry, boundry:-boundry] = np.mean(u[boundry:-boundry, boundry:-boundry])
            v[boundry:-boundry, boundry:-boundry] = np.mean(v[boundry:-boundry, boundry:-boundry])
            out_frame = warp_image(frame, prev_u + u, prev_v + v)

            # Fix border effects by cutting out a constant portion of the image
            out_frame = out_frame[start_rows:-end_rows, start_cols:-end_cols]
            if out_frame.T.shape != IMAGE_SIZE:
                out_frame = cv2.resize(out_frame, IMAGE_SIZE)

            if DEBUG:
                frame_num += 1
                path = os.path.join(FASTER_FIX_DIR, f'frame_{frame_num}.png')
                cv2.imwrite(path, out_frame)

            out.write(np.uint8(out_frame))
            pbar.update(1)

            prev_u, prev_v = prev_u + u, prev_v + v
            prev_frame = frame

        else:
            break

    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
