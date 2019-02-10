#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>
#include <iostream>
#define RED Scalar(0,0,255)

using namespace cv;
Mat boxFindImg(std::vector<DMatch> good_match, std::vector<KeyPoint> img1keypoint, std::vector<KeyPoint> camkeypoint, Mat img1, Mat finalOutputImg);

int main()
{
    Mat cam, img1;
    std::vector<KeyPoint> img1keypoint, camkeypoint;

    img1 = imread("./data1/opencv.png", IMREAD_GRAYSCALE);
    SIFT instance_FeatureDetector;
    instance_FeatureDetector.detect(img1, img1keypoint);

    VideoCapture cap;
    cap.open(0);

    if (!cap.isOpened())
    {
        std::cout << "카메라가 열리지 않습니다" << std::endl;
        return -1;
    }

    for (;;) {
        cap.read(cam);
        imshow("camera", cam);

        int c = waitKey(33);

        // esc 누르면 반복 종료
        if (c == 27)
            break;
        
        instance_FeatureDetector.detect(cam, camkeypoint);

        SIFT instance_Descriptor;
        Mat img1outputarray, camoutputarray;

        instance_Descriptor.compute(img1, img1keypoint, img1outputarray);
        instance_Descriptor.compute(cam, camkeypoint, camoutputarray);

        FlannBasedMatcher FLANNmatcher;
        std::vector<DMatch> match;

        FLANNmatcher.match(img1outputarray, camoutputarray, match);

        double maxd = 0; double mind = match[0].distance;

        for (int i = 0; i < match.size(); i++)
        {
            double dist = match[i].distance;

            if (dist < mind) mind = dist;
            if (dist > maxd) maxd = dist;
        }

        std::vector<DMatch> good_match;

        for (int i = 0; i < match.size(); i++)
        {
            if (match[i].distance <= max(2 * mind, 0.02)) good_match.push_back(match[i]);
        }

        Mat finalOutputImg;

        drawMatches(img1, img1keypoint, cam, camkeypoint, good_match, finalOutputImg, Scalar(150, 30, 200), Scalar(0, 0, 255), std::vector<char>(), DrawMatchesFlags::DEFAULT);
        finalOutputImg = boxFindImg(good_match, img1keypoint, camkeypoint, img1, finalOutputImg);
        imshow("매칭 결과", finalOutputImg);
    }

    return 0;
}

Mat boxFindImg(std::vector<DMatch> good_match, std::vector<KeyPoint> img1keypoint, std::vector<KeyPoint> camkeypoint, Mat img1, Mat finalOutputImg)
{
    std::vector<Point2f> model_pt;
    std::vector<Point2f> scene_pt;

    for (int i = 0; i < good_match.size(); i++) {
        model_pt.push_back(img1keypoint[good_match[i].queryIdx].pt);
        scene_pt.push_back(camkeypoint[good_match[i].trainIdx].pt);
    }

    Mat H = findHomography(model_pt, scene_pt, CV_RANSAC);

    std::vector<Point2f> model_corner(4);

    model_corner[0] = cvPoint(0, 0);
    model_corner[1] = cvPoint(img1.cols, 0);
    model_corner[2] = cvPoint(img1.cols, img1.rows);
    model_corner[3] = cvPoint(0, img1.rows);

    std::vector<Point2f> scene_corner(4);
    perspectiveTransform(model_corner, scene_corner, H);

    Point2f p(img1.cols, 0);

    line(finalOutputImg, scene_corner[0] + p, scene_corner[1] + p, RED, 3);
	line(finalOutputImg, scene_corner[1] + p, scene_corner[2] + p, RED, 3);
	line(finalOutputImg, scene_corner[2] + p, scene_corner[3] + p, RED, 3);
	line(finalOutputImg, scene_corner[3] + p, scene_corner[0] + p, RED, 3);

    return finalOutputImg;
}
