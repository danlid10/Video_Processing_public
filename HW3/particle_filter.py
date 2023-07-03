import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "42"
ID2 = "42"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y

# Noise parameters
MU = 0
POS_SIGMA = 1
VEL_SIGMA = 0.5

def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    """ DELETE THE LINE ABOVE AND:
    INSERT YOUR CODE HERE."""

    state_drifted = s_prior.copy()
    # Update X_c and Y_c from X_velocity and Y_velocity
    state_drifted[:2,:] += state_drifted[4:,:]  
    # Add noise
    state_drifted[0, :] += np.random.normal(MU, POS_SIGMA, N)   # X_center
    state_drifted[1, :] += np.random.normal(MU, POS_SIGMA, N)   # Y_center
    state_drifted[4, :] += np.random.normal(MU, VEL_SIGMA, N)   # X_velocity
    state_drifted[5, :] += np.random.normal(MU, VEL_SIGMA, N)   # Y_velocity
    state_drifted = np.round(state_drifted)

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    x, y, half_w, half_h, x_v, y_v = state
    cropped_image = image[y - half_h: y + half_h + 1, x - half_w: x + half_w + 1]
    b, g, r = cv2.split(cropped_image)

    # Quantize 16 bits to 4 bits
    b, g, r = b//16, g//16, r//16
    
    hist = np.zeros((16, 16, 16))
    for i in range(cropped_image.shape[0]):
         for j in range(cropped_image.shape[1]):
            hist[b[i, j], g[i, j], r[i, j]] += 1

    hist = np.reshape(hist, 16 * 16 * 16)

    # Normalize histogram
    hist = hist/sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cumulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    S_next = np.zeros_like(previous_state)
    for i in range(previous_state.shape[1]): 
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        S_next[:,i] = previous_state[:,j]

    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    distance = np.exp(20*np.sum(np.sqrt(p*q)))
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame number = " + str(frame_index))

    # Avg particle box
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
   
    x_avg = np.sum(state[0]*W) - state[2, 0]
    y_avg = np.sum(state[1]*W) - state[3, 0]
    w_avg = state[2, 0] * 2
    h_avg = state[3, 0] * 2
 
    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
 
    x_max = state[0, np.argmax(W)] - state[2, 0]
    y_max = state[1, np.argmax(W)] - state[3, 0]
    w_max = state[2, 0] * 2
    h_max = state[3, 0] * 2

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state

def compute_slacks_weights(image: np.ndarray, q: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the weights and the slacks (vector C).

    Args:
        image: np.ndarray. 
        q: np.ndarray. histogram
        state: np.ndarray. state, shape: (6, N)

    Return:
        slacks: np.ndarray. Slacks of each particle (vector C).
        weights: np.ndarray. Weights of each particle.
    """
    p_mat = [compute_normalized_histogram(image, col) for col in state.T]
    weights = [bhattacharyya_distance(p, q) for p in p_mat]
    weights = np.array(weights)
    weights /= np.sum(weights)
    slacks = np.cumsum(weights)
    return slacks, weights

def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""

    C, W = compute_slacks_weights(image, q, S)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""

        C, W = compute_slacks_weights(current_image, q, S)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
