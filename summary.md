# summary

## week1

- keywords: cv, ml, dl
- what is a cv? image(or video) -> sensing device(eye, camera) -> interpreting device(brain, cpu) -> interpretations(some information)
- what is ml/dl?
  - ml: 대량의 데이터로부터 지식이나 패턴을 찾아 학습하고 예측을 수행하는 것
- what is a visual recognition?
  - object identification
  - object classification
  - object detection (2D)
  - object detection (3D)

### feature based image stitching

- what is a stitching?
  - 비슷한 두 개 이상의 사진을 이어 붙이는 것
  1. detect feature points in both images
  2. find corresponding pairs
  3. use theses pairs to align images
- which feature(detector) is best?
  - detect the same point independently in both images
  - a repeatable detector
  - a reliable and distinctive descroptor
- image translations are not enough to align the images (such as panorama)
- transform
  - translation: 위치 이동, 2 points
  - affine: 변환 이동, 6 points
  - perspective: 8 points
- homograph
  - projective mapping between any two PPs with the same center of projection
- RANSAC: RANdom SAmple Consensus
    1. randomly choose a subset of data points to fit model
    2. points within some distance threshold t of model are a consensus set size of consensus set is model's support
    3. repeat for N samples; model with the biggest support is the most robust fit
      - points within distance t of best model are inliers
      - fit final model to all inliers

> image alignment and stitching: a tutorial, Richard Szeliski
> computer vision: algorithms and applications, Richard Szeliski, ch 4 6 9

- finding corners
  - in the region around a corner, image gradient has two or more dominant directions
  - flat
  - edge
  - corner
- Harris corner / Harris detector
  1. compute Gaussian derivatives at each pixel
  2. compute second moment matrix M in a Gaussian window around each pixel
  3. Compute corner response function R
  4. threshold R
  5. find local maxima of response function (nonmaximum suppression)
  - invariance
- models of image change
  - rotation
  - scale
  - affine
  - affine intensity change (photometric)
- scale-invariant feature detection
  - goal: independently detect corresponding regions in scaled versions of the same images
  - need scale selection mechanism for finding characteristic region size that is covariant with the image transformation
  1. convolve image with scale-normalized Laplacian at several scales
  2. find maxima of squared Laplacian response in scale-space
  - approximating the Laplacian with a difference of Gaussians
  - SSD, NCC
- which feature is best?
  - Feature = (repeatable) detector + (distinctive) descriptor

### SIFT

- Blob Detector -> (x, y, s)
- Scale invariance

1. create scale space
    - Gaussian pyramid를 통해 high resolution -> low resolution 으로 변환
2. create DoG (Difference of Gaussian)
    - scale space pyramid
    - maxima and minima of the scaled Laplacian provides the _most stable scale invariant features_
    - efficient to compute: smoothed images L needed later so D can be computed by simple image subtraction
3. extract DoG extrema
    - find local maxima across scale/space
    - a good "blob" detector
    - "only single-scale accepted"

- Blob Detector -> (o)
- Rotation invariance

4. assign orientations
    - compute gradient for each blurred image
    - for region around keypoint

- Blob Descriptor -> 128 dim (float)
- Photometric/Geometric invariance

5. build descriptors
    - create histogram for each sub region with 8 bins
    - 8 방향의 벡터로 구성, 4x4 descriptors, 4x4x8=128 element vector

### ORB

- Oriented FAST and Rotated BRIEF
- good alternative to SIFT and SURF in computation cost

1. FAST corner + Harris-cornerness
2. multi-scale keypoints
3. single-orientation keypoints
4. binary descriptor using "Binary Pattern Test"
5. trained Binary Pattern

## week2

- 대표적인 ml 문제
  - classfication or regression
  - supervised learning vs unsupervised learning
- classification: feature + classifier
  - feature: HoG, BoW, CNN, Haar-like
  - classifier: SVM, Random Forest, Adaboost Cascade Classifier

### HoG

- Histogram of Gradient

1. compute gradient (previous step: compute edge with kernel)
2. generate local histogram
3. concatenate local histogram -> 1D vector

- block 단위로는 기하학적 정보를 유지
- 각 block 내부에서는 histogram을 사용함으로써 local 변화에도 큰 문제가 없음
- edge 기반이라 밝기 변화, 조명 변화에 덜 민감
- 윤곽선을 찾는 것이므로 패턴이 복잡하지 않고 고유한 선 정보를 갖는 물체를 식별하는데 적합
- 회전, 형태 변화가 심하면 민감
- 1 block = 16 x 16 pixel, 8 pixel 씩 shift (= 64 x 128 pixel 이미지는 ((64/16)+3)block x ((128/16)+7)block = 7 block x 15 block)
- HoG descriptor length = #blocks x #cellsPerBlock x #binsPerCell = 7 x 15 x (2 x 2) x 9 (40 degree) = 3780 ndim

### CNN

- black box

### BoW

- image 조각들이 나눠져있고, 조각들이 목표가 되는 이미지와 일치하면 만족도가 높음

### SVM

- Support Vector Machine
- the margin of a linear classifier as the width that the boundary could be increased by before hitting a datapoint
- maximizing the margin
- underfit / robust / overfit
- soft margin classification: slack variables can be added to allow misclassification of difficult or noisy examples
- non-linear SVMs

### Haar feature

- image region의 밝기 차이를 이용
- white areas are subtracted from the black areas
- cascade 
  - 관심 대상 외의 영역들은 먼저 제외해서 찾고자 하는 대상과 유사한 영역에 대해 집중하기 위해 classifier를 cascade(직렬) 배치
  - 점점 검출에 쓰이는 영역이 작아지기 때문에 계산량이 줄고, 속도 상승

## week3

- object detection
  - assigning a label and a bounding box to all objects
  - finding a needle in a haystack (too many positions and scales to test)
- history (p9)
- TP, FP, FN, TN
  - precision@t = #TP@t / (#TP@t + #FP@t)
  - recall@t = #TP@t / #GT objects = #TP@t / (#TP@t + #FN@t)
- IoU
  - area of Intersection(overlap) / area of Union
- AP
  - Average Precision for each category
  - AP = sum_{r <- Recall([0,1])}^{Precision(t_r)} / |Recaal([0,1])|
  - mAP: average over classes
- HoG detector pipeline
    1. scan an image at all scales and locations: sliding window
    2. extract features over windows: compute gradients -> weighted vote into spatial & orientation cells -> contrast normalize over overlapping spatial blocks
    3. run linear SVM classifier onf all locations: train a classifier -> predict pos/neg of object in each window
    4. fuse multiple detections in position & scale space
- DPM detector
  - 신체의 부위에 스프링이 달려서 움직임에 보다 유연함
  - a mixture model consists of m components
  - captures extreme intra-class variation
  - split the positive bounding boxes into m groups by aspect ratio

### Renaissance

- CNN
  - Convolutional neural networks, ConvNet
  - a class of deep, feed-forward (not recurrent) artificial neural networks that the are applied to analyzing visual imagery
  - Conv
    - sparse interactions: input neuron, only connected to neighboring units
    - parameter sharing
    - equivariant representations
  - Pooling
    - subsamples feature maps
- what's new since the 1980s?
  - more layers
  - ReLU non-linearities
  - Dropout regularization
  - fast GPU implementations
  - more data

### Detection with CNN

- deep architecture: feature extractor + meta-architecture
  - feature extractor: VGG16, Inception, ResNet, Inception ResNet, MobileNet, ...
  - meta-architecture for object detection: R-CNN/Fast R-CNN/Faster R-CNN, OverFeat, YOLO, SSD, ...

### R-CNN

### Fast R-CNN

### Faster R-CNN

## week4
