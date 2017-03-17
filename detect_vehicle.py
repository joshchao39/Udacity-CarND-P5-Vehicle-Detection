import time

from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_operation import *

# Directories
OUT_IMG_DIR = 'output_images/'
TEST_IMG_DIR = 'test_images/'
TEST_OUTPUT_DIR = 'test_output/'
LARGE_FILES_DIR = 'large_files/'
VEHICLE_DIR = LARGE_FILES_DIR + 'vehicles/'
NON_VEHICLE_DIR = LARGE_FILES_DIR + 'non-vehicles/'

if __name__ == "__main__":
    start_time = time.time()

    """
    Extract features from training data set and generate scale normalizer
    """
    if False:
        # Identify all training image paths
        vehicle_image_paths = [path for d in list_directory(VEHICLE_DIR) for path in list_directory(d)]
        non_vehicle_image_paths = [path for d in list_directory(NON_VEHICLE_DIR) for path in list_directory(d)]
        # Extract features
        vehicle_features = extract_features_list(vehicle_image_paths)
        non_vehicle_features = extract_features_list(non_vehicle_image_paths)
        # Normalize features to ensure same scale across features
        scaler = StandardScaler().fit(np.vstack((vehicle_features, non_vehicle_features)))
        vehicle_features_scaled = scaler.transform(vehicle_features)
        non_vehicle_features_scaled = scaler.transform(non_vehicle_features)

        print('vehicle_features_scaled.shape', vehicle_features_scaled.shape)
        print('non_vehicle_features_scaled.shape', non_vehicle_features_scaled.shape)

        # Pickle the scaled features for faster retrieval
        joblib.dump(vehicle_features_scaled, 'vehicle_features.pickle')
        joblib.dump(non_vehicle_features_scaled, 'non_vehicle_features.pickle')
        joblib.dump(scaler, 'scaler.pkl')

    """
    Train a SVM classifier based on extracted features
    """
    if False:
        # Retrieve pickled feature lists
        vehicle_features_scaled = joblib.load('vehicle_features.pickle')
        non_vehicle_features_scaled = joblib.load('non_vehicle_features.pickle')

        # Generate labels for each class
        vehicle_labels = np.ones((vehicle_features_scaled.shape[0], 1))
        non_vehicle_labels = np.zeros((non_vehicle_features_scaled.shape[0], 1))
        scaled_x = np.vstack((vehicle_features_scaled, non_vehicle_features_scaled))
        y = np.vstack((vehicle_labels, non_vehicle_labels))

        # Split data into training and testing
        x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2)

        # Create a SVC (support vector classifier)
        svc = SVC(kernel='rbf', verbose=True)
        # Train the SVC
        svc.fit(x_train, y_train)
        print('Test Accuracy of SVC = ', svc.score(x_test, y_test))
        joblib.dump(svc, 'model.pkl')

    """Identify cars in test images"""
    # Retrieve pre-trained model
    FRAME_INFO_STORE.svc = joblib.load('model.pkl')
    FRAME_INFO_STORE.scaler = joblib.load('scaler.pkl')
    if True:
        for path in list_directory(TEST_IMG_DIR):
            print('path', path, flush=True)
            # Clear frame information store for test images processing
            FRAME_INFO_STORE.clear()
            img = read_image(path)
            # ret = detect_vehicle_boxes(img)
            # ret = detect_vehicle_heat_map(img)
            ret = detect_vehicle_aug(img)
            # ret_title = 'Image with windows drawn'
            # ret_title = 'Heat Map'
            ret_title = 'Augmented Image'
            save_2_images(img, ret, 'Original Image', ret_title,
                          TEST_OUTPUT_DIR + os.path.split(path)[-1].split('.')[0] + '.png')

    """Identify cars in videos"""
    if False:
        clip = VideoFileClip("test_video.mp4")
        processed_clip = clip.fl_image(detect_vehicle_boxes)
        processed_clip.write_videofile(OUT_IMG_DIR + 'test_video_boxes.mp4', audio=False)
        # clip = VideoFileClip("test_video.mp4")
        # processed_clip = clip.fl_image(detect_vehicle_heat_map)
        # processed_clip.write_videofile(OUT_IMG_DIR + 'test_video_heat_map.mp4', audio=False)
        # clip = VideoFileClip("test_video.mp4")
        # processed_clip = clip.fl_image(detect_vehicle_aug)
        # processed_clip.write_videofile(OUT_IMG_DIR + 'test_video_augmented.mp4', audio=False)
        # clip = VideoFileClip("project_video.mp4")
        # processed_clip = clip.fl_image(detect_vehicle_boxes)

        # processed_clip.write_videofile(OUT_IMG_DIR + 'project_video_boxes.mp4', audio=False)
        # clip = VideoFileClip("project_video.mp4")
        # processed_clip = clip.fl_image(detect_vehicle_heat_map)
        # processed_clip.write_videofile(OUT_IMG_DIR + 'project_video_heat_map.mp4', audio=False)
        # clip = VideoFileClip("project_video.mp4")
        # processed_clip = clip.fl_image(detect_vehicle_aug)
        # processed_clip.write_videofile(OUT_IMG_DIR + 'project_video_augmented.mp4', audio=False)

    print("--- {:.3f} seconds ---".format(time.time() - start_time))
