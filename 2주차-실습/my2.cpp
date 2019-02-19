#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectAndDisplay(Mat frame);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main()
{
    String face_cascade_name = "./data2/data/haarcascades/haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "./data2/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    int camera_device = 0;

    if (!face_cascade.load(face_cascade_name))
    {
        cout << "Error: loading face cascade" << endl;
        return -1;
    }
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "Error: loading eyes cascade" << endl;
        return -1;
    }

    VideoCapture capture;

    if (!capture.open(camera_device))
    {
        cout << "Error: open camera device" << endl;
        return -1;
    }

    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "Error: no captured frame" << endl;
            continue;
        }

        detectAndDisplay(frame);
        
        if (waitKey(33) == 27)
        {
            break;
        }
    }

    return 0;
}

void detectAndDisplay(Mat frame_origin)
{
    Mat frame;

    cvtColor(frame_origin, frame, COLOR_BGR2GRAY);
    equalizeHist(frame, frame);

    vector<Rect> faces;

    cout << "Detecting faces..." << endl;
    face_cascade.detectMultiScale(frame, faces);
    cout << "Finish!" << endl;

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame_origin, center, Size(faces[i].width / 2 , faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);

        Mat faceROI = frame(faces[i]);
        
        vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);

        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame_origin, eye_center, radius, Scalar(255, 0, 0), 4);
        }
    }

    imshow("Detected faces!", frame_origin);
}
