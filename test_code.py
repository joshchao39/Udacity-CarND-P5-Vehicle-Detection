from detect_vehicle import *
import matplotlib.pyplot as plt

"""Common"""
start_time = time.time()

"""Make sure images are in the same scale"""
if False:
    test_paths = list_directory(TEST_IMG_DIR)
    path = test_paths[random.randint(0, len(test_paths) - 1)]
    print('path', path)
    img = read_image(path)
    print(np.max(img))
    print(np.min(img))

    vehicle_image_paths = [path for d in list_directory(VEHICLE_DIR) for path in list_directory(d)]
    path = vehicle_image_paths[random.randint(0, len(vehicle_image_paths) - 1)]
    print('path', path)
    img = read_image(path)
    print(np.max(img))
    print(np.min(img))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

"""Visualize box coverage"""
if True:
    for path in list_directory(TEST_IMG_DIR):
        print('path', path)
        img = read_image(path)
        height, width, depth = img.shape
        img_drawn = np.copy(img)
        colors_iter = iter(COLORS)

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
            for window in window_list:
                cv2.rectangle(img_drawn, window[0], window[1], color=color, thickness=2)

        fig = plt.figure()
        plt.axis('off')
        plt.imshow(img_drawn)
        axes = fig.get_axes()
        for axis in axes:
            axis.set_axis_off()
        plt.savefig(TEST_OUTPUT_DIR + os.path.split(path)[-1].split('.')[0] + '.png', bbox_inches='tight')

        break

"""Test color map"""
if False:
    img = read_image(list_directory(TEST_IMG_DIR)[0])
    print('original')
    print(img.shape)
    print(img.dtype)
    print(np.min(img))
    print(np.max(img))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print('gray')
    print(gray.shape)
    print(gray.dtype)
    print(np.min(gray))
    print(np.max(gray))
    color = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    print('color')
    print(color.shape)
    print(color.dtype)
    print(np.min(color))
    print(np.max(color))
    save_2_images(gray, color, 'Gray', 'Heat Map', TEST_OUTPUT_DIR + 'test_heat_map' + '.png')

"""Draw Random Training Images"""
if False:
    vehicle_image_paths = [path for d in list_directory(VEHICLE_DIR) for path in list_directory(d)]
    non_vehicle_image_paths = [path for d in list_directory(NON_VEHICLE_DIR) for path in list_directory(d)]

    num_images = 8
    fig = plt.figure()
    plt.axis('off')
    for i in range(num_images):
        ii = random.randint(0, len(non_vehicle_image_paths))
        img = read_image(non_vehicle_image_paths[ii])
        fig.add_subplot(1, num_images, i + 1)
        plt.imshow(img)

    axes = fig.get_axes()
    for axis in axes:
        axis.set_axis_off()
    # plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0, wspace=0.05)
    plt.savefig(TEST_OUTPUT_DIR + 'non_vehicles.png', bbox_inches='tight')

print("--- %s seconds ---" % (time.time() - start_time))
