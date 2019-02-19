#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

#define NUM_train 450
#define NUM_test 50

using namespace cv;
using namespace std;

void ConvertVectortoMatrix(vector<vector<float>> &trainHOG, vector<vector<float>> &testHOG, Mat &trainMat, Mat &testMat);
vector<float> find_HOG_feature_image(Mat img);

int main()
{
    vector<vector<float>> HOG_train_data,  HOG_test_data;
	vector<int> HOG_train_data_label, HOG_test_data_label;

    cout << "\n---------------------------------" << endl;
    cout << "train Image load" << endl;

    for (int num = 0; num < 10; num++)
    {
        cout << num << " image load" << endl;
        
        for (int i = 0; i < NUM_train; i++)
        {
            Mat MNIST = imread("./data1/MNIST/data " + to_string(num) + "_" + to_string(i + 1) + ".PNG", COLOR_RGB2GRAY);
            if (!(MNIST.data))
            {
                cout << "image load fail" << endl;
                return 0;
            }
            HOG_train_data.push_back(find_HOG_feature_image(MNIST));
            HOG_train_data_label.push_back(num);
        }
    }
    cout << "\n---------------------------------" << endl;
    cout << "test Image load" << endl;

    for (int num = 0; num < 10; num++)
    {
        cout << num << " image load" << endl;
        
        for (int i = NUM_train; i < (NUM_train + NUM_test); i++)
        {
            Mat MNIST = imread("./data1/MNIST/data " + to_string(num) + "_" + to_string(i + 1) + ".PNG", COLOR_RGB2GRAY);
            if (!(MNIST.data))
            {
                cout << "image load fail" << endl;
                return 0;
            }
            HOG_test_data.push_back(find_HOG_feature_image(MNIST));
            HOG_test_data_label.push_back(num);
        }
    }
    cout << "\n---------------------------------" << endl;

    int descriptor_size = HOG_train_data[0].size();

    Mat HOG_train_data_Mat(HOG_train_data.size(), descriptor_size, CV_32FC1);
    Mat HOG_test_data_Mat(HOG_test_data.size(), descriptor_size, CV_32FC1);
    Mat HOG_train_data_label_Mat(HOG_train_data_label.size(), 1, CV_32FC1);
    Mat HOG_test_data_label_Mat(HOG_test_data_label.size(), 1, CV_32FC1);

    ConvertVectortoMatrix(HOG_train_data, HOG_test_data, HOG_train_data_Mat, HOG_test_data_Mat);

    for (int i = 0; i < HOG_train_data_label.size(); i++)
    {
        HOG_train_data_label_Mat.at<float>(i, 0) = HOG_train_data_label[i];
    }
    for (int i = 0; i < HOG_test_data_label.size(); i++)
    {
        HOG_test_data_label_Mat.at<float>(i, 0) = HOG_test_data_label[i];
    }

    cout << "\n---------------------------------" << endl;
    cout << "               svm setting         " << endl;
    cout << "---------------------------------" << endl;

    CvSVM svm;

    CvSVMParams params = CvSVMParams(
        CvSVM::C_SVC,   // Type of SVM
        CvSVM::LINEAR,  // kernel type
        0.0,            // kernel parameter, degree
        0.0,            // kernel parameter, gamma
        0.0,            // kernel parameter, coef0
        10,             // SVM optimization parameter C
        0,              // SVM optimization parameter nu
        0,              // SVM optimization parameter p
        NULL,           // class weights
        cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)
    );

    // cout << "\n---------------------------------" << endl;
    // cout << "               svm train           " << endl;
    // cout << "---------------------------------" << endl;

    // svm.train_auto(HOG_train_data_Mat, HOG_train_data_label_Mat, Mat(), Mat(), params, 10);

    // cout << "\n---------------------------------" << endl;
    // cout << "               svm save            " << endl;
    // cout << "---------------------------------" << endl;

    // svm.save("./data1/MNIST_HOG_SVM.xml");

    cout << "\n---------------------------------" << endl;
    cout << "               svm load         " << endl;
    cout << "---------------------------------" << endl;

    svm.load("./data1/MNIST_HOG_SVM_Linear_acc97.xml");

    cout << "\n---------------------------------" << endl;
    cout << "               svm predict         " << endl;
    cout << "---------------------------------" << endl;

    Mat Response_train;
    Mat Response_test;

    svm.predict(HOG_train_data_Mat, Response_train);
    float count_train = 0, accuracy_train = 0;

    for (int i = 0; i < Response_train.rows; i++)
    {
        if (Response_train.at<float>(i, 0) == HOG_train_data_label[i])
        {
            count_train = count_train + 1;
        }
    }

    svm.predict(HOG_test_data_Mat, Response_test);
    float count_test = 0, accuracy_test = 0;

    for (int i = 0; i < Response_test.rows; i++)
    {
        if (Response_test.at<float>(i, 0) == HOG_test_data_label[i])
        {
            count_test = count_test + 1;
        }
    }

    accuracy_train = (count_train / Response_train.rows) * 100;
    cout << "accuracy_train : " << accuracy_train << endl;
    accuracy_test = (count_test / Response_test.rows) * 100;
    cout << "accuracy_test : " << accuracy_test << endl;
}

void ConvertVectortoMatrix(vector<vector<float>> &trainHOG, vector<vector<float>> &testHOG, Mat &trainMat, Mat &testMat)
{
    int descriptor_size = trainHOG[0].size();

    for (int i = 0; i < trainHOG.size(); i++)
    {
        for (int j = 0; j < descriptor_size; j++)
        {
            trainMat.at<float>(i, j) = trainHOG[i][j];
        }
    }

    for (int i = 0; i < testHOG.size(); i++)
    {
        for (int j = 0; j < descriptor_size; j++)
        {
            testMat.at<float>(i, j) = testHOG[i][j];
        }
    }
}

vector<float> find_HOG_feature_image(Mat img)
{
    HOGDescriptor IMAGE_HOG
    {
        Size(20, 20), //winSize
        Size(8, 8),   //blockSize
        Size(4, 4),   //blockStride
        Size(8, 8),   //cellSize,
        9,            //nbins,
        1,            //derivAper,
        -1,           //winSigma,
        0,            //histogramNormType,
        0.2,          //L2HysThresh,
        0,            //gammaLCorrection,
        64            //nlevels
    };

    vector<float> hog_descriptor;
    IMAGE_HOG.compute(img, hog_descriptor);

    return hog_descriptor;
}
