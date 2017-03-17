# Vehicle Detection


[//]: # (Image References)
[image1]: ./output_images/vehicles.png
[image2]: ./output_images/non_vehicles.png
[image3]: ./output_images/slide_windows.png
[image4]: ./output_images/slide_windows_80.png
[image5]: ./output_images/detected_windows.png
[image6]: ./output_images/heat_map.png
[image7]: ./output_images/bboxes.png

### Project Structure
* `detect_vehicle.py` script containing procedures to generate processed images and video  
* `image_operation.py` module containing all essential code to process images and video  
* `FrameInfoStore.py` class containing data to be shared between frames in video  
* `utils.py` module containing basic utility functions
* `test_code.py` scripts containing code to troubleshoot/experiment the system
* `README.md` summary detailing the steps and the results

### Project Breakdown

The approach employed by this project to detect vehicle can be broken down into the following steps:
1. Train an binary classifier to determine whether an image window contains a car or not
2. Grid search the video frames using windows with different sizes to generate a heat map corresponding to the likelihood of the presence of a vehicle
3. Use spatial and temporal filters to remove noise
4. Draw bounding boxes around where vehicles are likely to locate


### Train binary classifier

#### Training data
8,792 vehicle images and 8,968 non-vehicle images were provided to train the binary classifier.

Sample images with vehicles:

![alt text][image1]

Sample images without vehicles:

![alt text][image2]

#### Extract features
Two types of features were extracted from the training images:
- **HOG (Histogram of Oriented Gradients)** features
  - More information about HOG can be found [here](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients).
  - Parameters tuned:
    1. Orient - 9 was used
    2. Pixels per cell - 8 was used
    3. Cell per block - 2 was used
  - In general the higher the number the more information the classifier can use to improve accuracy, at the expense of training and serving speed.
  - HOG provides edge information (first order derivative)
- **Color binning** features.
  - Color binning is simply down-sampling the image and flattening the pixel values as features.
  - Parameter Tune:
    1. Window size - 16x16 was used
  - As the HOG parameters, 16x16 was used to balance the accuracy and speed.
  - Color binning provides pixel color information

Overall these two feature sets complement each other well, and they are concatenated to form the features for each image.

#### Training
Since the class distribution are about even no redistribution was necessary. The data set was then split 80/20 randomly for training/testing.
Several SVM kernels were used to compare the performance:

| Kernel  | Test Accuracy |
|:-------:|:-------------:| 
| linear  | 97.41%        |
| poly    | 99.13%        |
| **rbf** | **99.30%**    |
| sigmoid | 93.13%        |

Since we can already achieve 99%+ test accuracy using `rbf` kernel and we will be able to use spatial and temporal filters later on to remove more false positive, this result is good enough.  


### Slide detection window
Since the binary classifier was not trained to pinpoint the location and size of the vehicle, we need to slide and resize our detection window around each frame.

Just like any exhaustive search, this process is very slow, so we will need to limit the search to area where we expect the vehicle to be. We know cars further away appear smaller, and the no car should be in the sky (the upper half of the image).  

Window coverage demo 1 (No overlap):
![alt text][image3]

Note the above image was drawn with no overlap between windows for clarity. At serving I used 80% overlap to increase coverage:
![alt text][image4]

Combining with the binary classifier gets us the windows that has car detected:
![alt text][image5]

### Generate heat map
For each detected window, we add a score of 10 to the covered area. The more detection happens in an area, the higher the score the area gets. Using this technique we get a heat map for each frame:
![alt text][image6]

This heat map represents the likelihood of car presence in the spatial domain. We can then run the heat maps from successive frames through an exponential moving averager to extract the likelihood covering both spatial and temporal domains.

### Draw bounding boxes
With the heat map, we have the information to throw bounding boxes around cars, after some filtering:
1. Scores less than 5 are filtered out
2. For each heat "island", draw the largest rectangle contained by the covered area
3. Rectangles with less than 4096 pixels are filtered out

![alt text][image7]

### Video
To make the video more fun, I also applied the augmentation from the [Lane Finding project](https://github.com/joshchao39/Udacity-CarND-Lane-Finding-P4)! (Can't have enough augmentation)

<p align="center">
  <img src="./output_images/project_video_both_augmented.gif" alt="Augmented Driving Video"/>
</p>

Here's the [full video](https://youtu.be/QieNG8eA4Kk)

---

### Discussion
This approach works well for this video, but it is far from being the most efficient and accurate. To improve efficiency we can design a hierarchical classifier to quickly reject non-car windows. To improve accuracy we can use data augmentation to further generalize the classifier. There are many improvements that can be made to enhance the approach discussed in this report.

However, the biggest (smallest?) bottleneck is the slide window approach. Exhaustive search is rarely the best way to go in algorithms. Animals pay more attention to moving object, and perhaps we can expand on that idea to only run the classifier on area of the frame where movement is detected. (optical flow, motion history image, etc.)