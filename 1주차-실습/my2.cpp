#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#define NUM 3

using namespace cv;

Mat MakePano(Mat *imgArray, int num);

int main()
{
    Mat result;
    Mat imgArray[NUM];

    printf("input images...\n");
    for (int i = 0; i < NUM; i++)
    {
        imgArray[i] = imread("./data2/" + std::to_string(i + 1) + ".jpg");
    }

    printf("Panorama stitching func start...\n");
    result = MakePano(imgArray, NUM);
    printf("func finished!\n");

    printf("shows image...\n");
    imshow("result", result);

    waitKey();
    return 0;
}

Mat MakePano(Mat *imgArray, int num)
{
    Mat mainPano = imgArray[0];

    for (int i = 1; i < num; i++)
    {
        Mat gray_mainImg, gray_objImg;

        cvtColor(mainPano, gray_mainImg, COLOR_RGB2GRAY);
        cvtColor(imgArray[i], gray_objImg, COLOR_RGB2GRAY);

        SurfFeatureDetector detector(0.3);

        vector<KeyPoint> point1, point2;

        detector.detect(gray_mainImg, point1);
        detector.detect(gray_objImg, point2);

        SurfDescriptorExtractor extractor;
        Mat descriptor1, descriptor2;

        extractor.compute(gray_mainImg, point1, descriptor1);
        extractor.compute(gray_objImg, point2, descriptor2);

        printf("match keypoints...\n");
        FlannBasedMatcher matcher;
        vector<DMatch> matches;

        matcher.match(descriptor1, descriptor2, matches);

        printf("get minimal distance...\n");
        double mindistance = matches[0].distance;
        double distance;

        for (int i = 0; i < descriptor1.rows; i++)
        {
            distance = matches[i].distance;
            if (mindistance > distance) mindistance = distance;
        }

        printf("filtering mathes using minimal distance...\n");
        vector<DMatch> goodmatch;

        for (int i = 0; i < descriptor1.rows; i++)
        {
            if (matches[i].distance < 5 * mindistance) goodmatch.push_back(matches[i]);
        }

        printf("drawing lines between matched keypoints...\n");
        Mat matGoodMatches;
        drawMatches(mainPano, point1, imgArray[i], point2, goodmatch, matGoodMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        printf("(option)shows the image...\n");
        imshow("after drawing matches " + std::to_string(i), matGoodMatches);

        vector<Point2f> obj;
        vector<Point2f> scene;

        for (int i = 0; i < goodmatch.size(); i++)
        {
            obj.push_back(point1[goodmatch[i].queryIdx].pt);
            scene.push_back(point2[goodmatch[i].trainIdx].pt);
        }

        Mat homomatrix = findHomography(scene, obj, CV_RANSAC);
        Mat warp;

        warpPerspective(imgArray[i], warp, homomatrix, Size(imgArray[i].cols + mainPano.cols, imgArray[i].rows), INTER_CUBIC);

        Mat matPanorama;

        matPanorama = warp.clone();

        Mat matROI(matPanorama, Rect(0, 0, mainPano.cols, mainPano.rows));
        
        mainPano.copyTo(matROI);

        int max = 0;

        for (int i = 0; i < matPanorama.cols; i++)
        {
            if (matPanorama.at<Vec3b>(matPanorama.rows / 2, i) != Vec3b(0, 0, 0))
            {
                if (max < i) max = i;
            }
            Mat img = matPanorama(Range(0, matPanorama.rows), Range(0, max));
            mainPano = img;
        }

        imshow("after stitching " + std::to_string(i), mainPano);
    }
    
    return mainPano;
}