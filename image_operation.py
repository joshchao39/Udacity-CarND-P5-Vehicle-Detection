import itertools
import random
from multiprocessing import Pool

import cv2
from scipy.ndimage.measurements import label
from skimage.feature import hog

from FrameInfoStore import FrameInfoStore
from utils import *

# Color
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
COLORS = (RED, BLUE, GREEN, YELLOW, WHITE)

# Feature Extraction
INPUT_SIZE = (64, 64)

# Spatial Binning
BIN_WINDOW_SIZE = 16

# HOG
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2

# Car window detection
MIN_DETECTION_SIZE = 64 * 64
HEAT_THRESHOLD = 5

# Inter-frame information store
FRAME_INFO_STORE = FrameInfoStore()


def get_binning_features(img, size=(BIN_WINDOW_SIZE, BIN_WINDOW_SIZE)):
    """
    Generate features by binning pixel values
    """
    return cv2.resize(img, size).ravel()


def get_hog_features(img, vis=False, feature_vec=True):
    """
    Generate HOG (histogram of oriented gradient) features
    """
    if vis:
        # Use skimage.hog() to get both features and a visualization
        return hog(img, orientations=ORIENT, pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
                   cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK), visualise=vis, feature_vector=feature_vec)
    else:
        # Use skimage.hog() to get features only
        return hog(img, orientations=ORIENT, pixels_per_cell=(PIX_PER_CELL, PIX_PER_CELL),
                   cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK), visualise=vis, feature_vector=feature_vec)


def extract_features(img, gray, fast_filter=False):
    """
    Extract binning and HOG features from an image.
    :param img: Image window in color
    :param gray: Image window in gray-scale
    :param fast_filter: Enable the filter to quickly dis-quality a frame with little information (Use in serving only)
    """
    img_resize = cv2.resize(img, INPUT_SIZE)
    gray_resized = cv2.resize(gray, INPUT_SIZE)
    if fast_filter and np.max(gray) - np.min(gray) < 50:
        return []
    # Extract color binning features
    bin_features = get_binning_features(img_resize)
    # Extract HOG features
    hog_features = get_hog_features(gray_resized, vis=False, feature_vec=True)
    # Combine both feature sets
    features = np.concatenate((bin_features, hog_features))
    return features


def extract_features_list(img_paths):
    """
    Generate a list of features from an iterable of image paths
    """
    features_list = []
    for path in img_paths:
        # Read in image
        img = read_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features = extract_features(img, gray)
        # Collect across all images
        features_list.append(features)
    return np.array(features_list)


def slide_window(x_start_stop=None, y_start_stop=None, window_size=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Generate a list of windows by specifying start/stop, window size, and overlapping ratio
    """
    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    xy_step = (window_size[0] * (1 - xy_overlap[0]), window_size[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    x_buffer = window_size[0] * (xy_overlap[0])
    y_buffer = window_size[1] * (xy_overlap[1])
    xy_num_windows = ((x_span - x_buffer) / xy_step[0], (y_span - y_buffer) / xy_step[1])
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for window_i_x in range(int(xy_num_windows[0])):
        for window_i_y in range(int(xy_num_windows[1])):
            # Calculate each window position
            window_x_start = int(window_i_x * xy_step[0] + x_start_stop[0])
            window_x_end = window_x_start + int(window_size[0])
            window_y_start = int(window_i_y * xy_step[1] + y_start_stop[0])
            window_y_end = window_y_start + int(window_size[1])
            corner_1 = (window_x_start, window_y_start)
            corner_2 = (window_x_end, window_y_end)
            # Append window position to list
            window_list.append((corner_1, corner_2))
    # Return the list of windows
    return window_list


def is_car_image(args):
    """
    Determine whether the image is a car image using binary classifier
    :param args: 3-tuple of window size, color image, and gray image. Combined to allow multi-processing.
    :return: whether the image is a car (True) or not (False)
    """
    window = args[0]
    img = args[1]
    gray = args[2]
    window_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    window_gray = gray[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    features = extract_features(window_img, window_gray, fast_filter=True)
    return len(features) > 0 and FRAME_INFO_STORE.svc.predict(FRAME_INFO_STORE.scaler.transform(features))[0] == 1.0


def detect_vehicle(img, draw_box=False):
    """
    Identify all the locations in the image where a car is present
    """
    height, width, depth = img.shape
    img_detected = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_heat_map = np.zeros(img.shape[0:2])
    colors_iter = iter(COLORS)

    with Pool() as pool:
        for w_size in np.arange(64, 129, 32):
            # Narrow the search height for smaller window due to distance perspective
            offset = 288 - w_size * 2
            # Randomly shifts the windows slightly in each frame to reduce variance
            random_offset_x = random.randint(0, 20)
            random_offset_y = random.randint(12, 32)
            # Use different color for different window size
            color = next(colors_iter)
            # Generate a list of windows to search for car
            window_list = slide_window(x_start_stop=(random_offset_x, width),
                                       y_start_stop=(height / 2 + random_offset_y, height - offset),
                                       window_size=(w_size, w_size), xy_overlap=(0.8, 0.8))

            # Identify windows being a car using multi-processing
            inputs = zip(window_list, itertools.repeat(img), itertools.repeat(gray))
            valid_indices = pool.map(is_car_image, inputs)
            valid_windows = itertools.compress(window_list, valid_indices)
            for window in valid_windows:
                if draw_box:
                    cv2.rectangle(img_detected, window[0], window[1], color=color, thickness=2)
                new_heat_map[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 10

    return img_detected, new_heat_map


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = np.equal(labels[0], car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if len(nonzeroy) > MIN_DETECTION_SIZE:
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color=BLUE, thickness=4)
    # Return the image with bounding boxes drawn
    return img


def detect_vehicle_boxes(img):
    img_detected, new_heat_map = detect_vehicle(img, draw_box=True)
    return img_detected


def detect_vehicle_heat_map(img):
    img_detected, heat_map = detect_vehicle(img, draw_box=True)
    color = cv2.cvtColor(cv2.applyColorMap(heat_map.astype(np.uint8), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
    return color


def detect_vehicle_aug(img):
    img_detected, new_heat_map = detect_vehicle(img, draw_box=True)
    FRAME_INFO_STORE.update(new_heat_map)
    filtered_heat_map = apply_threshold(FRAME_INFO_STORE.heat_map, HEAT_THRESHOLD)
    labels = label(filtered_heat_map)
    img_aug = img.copy()
    draw_labeled_bboxes(img_aug, labels)
    return img_aug
