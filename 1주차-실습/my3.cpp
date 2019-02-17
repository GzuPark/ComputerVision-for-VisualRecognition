#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

typedef struct Image_feature
{
    Mat descriptors;
    vector<KeyPoint> keypoint;
    Mat img;
}Image_feature;

Image_feature find_image_feature(Mat img)
{
    vector<KeyPoint> keypoint;
    Image_feature img_class;
    Mat img_g;
    Mat descriptors;

    cvtColor(img, img_g, COLOR_BGR2GRAY );

    SiftDescriptorExtractor detector;
    detector.detect(img_g, keypoint);
    
    SiftDescriptorExtractor extractor;
    extractor.compute(img_g, keypoint, descriptors);

    img_class.descriptors = descriptors;
    img_class.keypoint = keypoint;
    img_class.img = img;

    return img_class;
}

double find_matches_percent(Image_feature img1, Image_feature img2)
{
    vector<DMatch> good_matches;
    vector<DMatch> matches;

    FlannBasedMatcher matcher;
    matcher.match(img1.descriptors, img2.descriptors, matches);

    double maxDistance = 0;
    double minDistance = 100;
    double distance;

    for (int i = 0; i < img1.descriptors.rows; i++)
    {
        distance = matches[i].distance;

        if (distance < minDistance) minDistance = distance;
        if (distance > maxDistance) maxDistance = distance;
    }

    for (int i = 0; i < img1.descriptors.rows; i++)
    {
        if (matches[i].distance < 3 * minDistance)
        {
            good_matches.push_back(matches[i]);
        }
    }

    vector<Point2f> img1_pt;
    vector<Point2f> img2_pt;

    for (int i = 0; good_matches.size(); i++)
    {
        img1_pt.push_back(img1.keypoint[good_matches[i].queryIdx].pt);
        img2_pt.push_back(img2.keypoint[good_matches[i].trainIdx].pt);
    }
    if (good_matches.size() == 0) return 0;

    Mat mask;
    Mat HomoMatrix = findHomography(img2_pt, img1_pt, RANSAC, 3, mask);

    double outline_cnt = 0;
    double inline_cnt = 0;

    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<bool>(i) == 0)
        {
            outline_cnt++;
        }
        else
        {
            inline_cnt++;
        }
    }
    double percentage = ((inline_cnt) / (inline_cnt + outline_cnt)) * 100;

    return percentage;
}

Mat panorama_stiching(Image_feature img1, Image_feature img2)
{
    vector<DMatch> good_matches;
    vector< vector<DMatch> > matches;

    BFMatcher matcher;
    matcher.knnMatch(img1.descriptors, img2.descriptors, matches, 2);

    for (int i = 0; i < matches.size(); i++)
    {
        const float ratio = 0.7;
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }
    cout << "Good match : " << good_matches.size() << endl;

    vector<Point2f> img1_pt;
    vector<Point2f> img2_pt;

    for (int i = 0; good_matches.size(); i++)
    {
        img1_pt.push_back(img1.keypoint[good_matches[i].queryIdx].pt);
        img2_pt.push_back(img2.keypoint[good_matches[i].trainIdx].pt);
    }
    Mat HomoMatrix = findHomography(img2_pt, img1_pt, RANSAC, 3);

    cout << HomoMatrix << endl;

    Mat matResult;
    Mat matPanorama;

    vector<Point2f> cornerPt;

    cornerPt.push_back(Point2f(0, 0));
    cornerPt.push_back(Point2f(img2.img.size().width, 0));
    cornerPt.push_back(Point2f(0, img2.img.size().height));
    cornerPt.push_back(Point2f(img2.img.size().width, img2.img.size().height));

    Mat perspectiveTransCornerPt;
    perspectiveTransform(Mat(cornerPt), perspectiveTransCornerPt, HomoMatrix);

    double minX, minY, maxX, maxY;
    float minX1, minX2, minY1, minY2, maxX1, maxX2, maxY1, maxY2;

    minX1 = min(perspectiveTransCornerPt.at<Point2f>(0).x, perspectiveTransCornerPt.at<Point2f>(1).x);
    minX2 = min(perspectiveTransCornerPt.at<Point2f>(2).x, perspectiveTransCornerPt.at<Point2f>(3).x);
    minY1 = min(perspectiveTransCornerPt.at<Point2f>(0).y, perspectiveTransCornerPt.at<Point2f>(1).y);
    minY2 = min(perspectiveTransCornerPt.at<Point2f>(2).y, perspectiveTransCornerPt.at<Point2f>(3).y);
    maxX1 = max(perspectiveTransCornerPt.at<Point2f>(0).x, perspectiveTransCornerPt.at<Point2f>(1).x);
    maxX2 = max(perspectiveTransCornerPt.at<Point2f>(2).x, perspectiveTransCornerPt.at<Point2f>(3).x);
    maxY1 = max(perspectiveTransCornerPt.at<Point2f>(0).y, perspectiveTransCornerPt.at<Point2f>(1).y);
    maxY2 = max(perspectiveTransCornerPt.at<Point2f>(2).y, perspectiveTransCornerPt.at<Point2f>(3).y);
    minX = min(minX1, minX2);
    minY = min(minY1, minY2);
    maxX = max(maxX1, maxX2);
    maxY = max(maxY1, maxY2);

    Mat Htr = Mat::eye(3, 3, CV_64F);

    if (minX < 0)
    {
        maxX = img1.img.size().width - minX;
        Htr.at<double>(0, 2) = -minX;
    }
    else
    {
        if (maxX < img1.img.size().width) maxX = img1.img.size().width;
    }

    if (minY < 0)
    {
        maxY = img1.img.size().height - minY;
        Htr.at<double>(1, 2) = -minY;
    }
    else
    {
        if (maxY < img1.img.size().height) maxY = img1.img.size().height;
    }
    matPanorama = Mat(Size(maxX, maxY), CV_32F);
    warpPerspective(img1.img, matPanorama, Htr, matPanorama.size(), INTER_CUBIC, BORDER_CONSTANT, 0);
    warpPerspective(img2.img, matPanorama, (Htr * HomoMatrix), matPanorama.size(), INTER_CUBIC, BORDER_CONSTANT, 0);

    return matPanorama;
}

int main()
{
    vector<Mat> panorama;
    vector<Image_feature> Image_array;
    Image_feature Image_feature;

    String folderpath = "./data3";
    vector<String> filenames;
    glob(folderpath, filenames);

    cout << "\n------- file load -------\n" << endl;

    for (size_t i = 0; i < filenames.size(); i++)
    {
        panorama.push_back(imread(filenames[i], IMREAD_COLOR));
        cout << filenames[i] << " load" << endl;
    }

    for (int i = 0; i < filenames.size(); i++)
    {
        Image_feature = find_image_feature(panorama[i]);
        Image_array.push_back(Image_feature);
        cout << filenames[i] << " finish" << endl;
    }

    vector<int> image_match_count;
    int match_count = 0;
    int bef_match_count = 0;
    int max_match = 0;
    int max_count = 0;

    for (int i = 0; i < filenames.size(); i++)
    {
        for (int j = 0; j < filenames.size(); j++)
        {
            if (i != j)
            {
                if (find_matches_percent(Image_array[i], Image_array[j]) >= 10) match_count++;
            }
        }
        if (max_count < match_count)
        {
            max_match = i;
            max_count = match_count;
        }
        cout << filenames[i] << " be matching " << match_count << " images" << endl;

        match_count = 0;
    }
    cout << "---" << max_match << endl;

    Image_array.erase(Image_array.begin() + max_match);

    Mat Panorama = imread(filenames[max_match], IMREAD_COLOR);
    double match = 0;
    int maxj = 0;
    double maxper = 0;

    for (int i = 0; i < filenames.size(); i++)
    {
        Image_feature = find_image_feature(Panorama);

        for (int j = 0; j < Image_array.size(); j++)
        {
            match = find_matches_percent(Image_feature, Image_array[j]);

            if (match > maxper)
            {
                maxper = match;
                maxj = j;
            }
        }
        cout << "---max match done --- " << maxj << endl;

        Panorama = panorama_stiching(Image_feature, Image_array[maxj]);
        cout << maxj << endl;

        Image_array.erase(Image_array.begin() + maxj);
        namedWindow("test", CV_WINDOW_FREERATIO);
        imshow("test", Panorama);
        waitKey(20);
        maxper = 0;
    }
    namedWindow("result", CV_WINDOW_FREERATIO);
    imshow("result", Panorama);
    imwrite("result.jpg", Panorama);
    waitKey(0);    

    return 0;
}
