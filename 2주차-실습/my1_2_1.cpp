#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

vector<vector<float>> CsvtoVector(String filename);
void ConvertVectortoMatrix(vector<vector<float>> &In_vector, Mat &Out_Mat);

int main()
{
    vector<vector<float>> CNN_train_data_label = CsvtoVector("./data1/image_train_label_rs.csv");
    vector<vector<float>> CNN_train_data = CsvtoVector("./data1/image_train_rs.csv");
    vector<vector<float>> CNN_test_data_label = CsvtoVector("./data1/image_test_label_rs.csv");
    vector<vector<float>> CNN_test_data = CsvtoVector("./data1/image_test_rs.csv");

    int descriptor_size = CNN_train_data[0].size();

    Mat MNIST_CNN_train_data_label_Mat(CNN_train_data_label.size(), 1, CV_32FC1);
    Mat MNIST_CNN_train_data_Mat(CNN_train_data.size(), descriptor_size, CV_32FC1);
    Mat MNIST_CNN_test_data_label_Mat(CNN_test_data_label.size(), 1, CV_32FC1);
    Mat MNIST_CNN_test_data_Mat(CNN_test_data.size(), descriptor_size, CV_32FC1);

    ConvertVectortoMatrix(CNN_train_data, MNIST_CNN_train_data_Mat);
    ConvertVectortoMatrix(CNN_test_data, MNIST_CNN_test_data_Mat);
    
    for (int i = 0; i < CNN_train_data_label.size(); i++)
    {
        MNIST_CNN_train_data_label_Mat.at<float>(i, 0) = CNN_train_data_label[i][0];
    }
    for (int i = 0; i < CNN_test_data_label.size(); i++)
    {
        MNIST_CNN_test_data_label_Mat.at<float>(i, 0) = CNN_test_data_label[i][0];
    }

    cout << "\n---------------------------------" << endl;
    cout << "               svm setting         " << endl;
    cout << "---------------------------------" << endl;

    CvSVM svm;

    CvSVMParams params = CvSVMParams
    (
        CvSVM::C_SVC,   // type of svm
        CvSVM::LINEAR,  // kernel type
        0.0,            // kernel parameter, degree
        0.0,            // kernel parameter, gamma
        0.0,            // kernel parameter, coef0
        10,             // svm optimization parameter C
        0,              // svm optimization parameter nu
        0,              // svm optimization parameter p
        NULL,           // class weights
        cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)
    );

    cout << "\n---------------------------------" << endl;
    cout << "               svm train           " << endl;
    cout << "---------------------------------" << endl;

    svm.train_auto(MNIST_CNN_train_data_Mat, MNIST_CNN_train_data_label_Mat, Mat(), Mat(), params, 10);

    cout << "\n---------------------------------" << endl;
    cout << "               svm save            " << endl;
    cout << "---------------------------------" << endl;

    svm.save("./data1/MNIST_CNN_SVM.xml");

    // cout << "\n---------------------------------" << endl;
    // cout << "               svm load         " << endl;
    // cout << "---------------------------------" << endl;

    // svm.load("./data1/MNIST_CNN_SVM_Linear_acc94.5.xml");

    cout << "\n---------------------------------" << endl;
    cout << "               svm predict         " << endl;
    cout << "---------------------------------" << endl;

    Mat Response_train;
    Mat Response_test;

    svm.predict(MNIST_CNN_train_data_Mat, Response_train);
    float count_train = 0, accuracy_train = 0;

    for (int i = 0; i < Response_train.rows; i++)
    {
        if (Response_train.at<float>(i, 0) == CNN_train_data_label[i][0])
        {
            count_train++;
        }
    }

    svm.predict(MNIST_CNN_test_data_Mat, Response_test);
    float count_test = 0, accuracy_test = 0;

    for (int i = 0; i < Response_test.rows; i++)
    {
        if (Response_test.at<float>(i, 0) == CNN_test_data_label[i][0])
        {
            count_test++;
        }
    }

    accuracy_train = (count_train / Response_train.rows) * 100;
    cout << "accuracy_train : " << accuracy_train << endl;
    accuracy_test = (count_test / Response_test.rows) * 100;
    cout << "accuracy_test : " << accuracy_test << endl;

    return 0;
}

vector<vector<float>> CsvtoVector(String filename)
{
    ifstream train_data(filename);
    vector<vector<float>> vector_data;

    if (!train_data.is_open())
    {
        cout << "Error: File open" << endl;
    }
    else
    {
        cout << "Find: " << filename << endl;
        cout << "\tstarting loading" << endl;
    }
    
    if (train_data)
    {
        int cnt = 0;
        string line;

        while (getline(train_data, line))
        {
            vector<float> train_data_vector_2;
            stringstream linStream(line);
            string cell;
            cnt++;

            while (getline(linStream, cell, ','))
            {
                train_data_vector_2.push_back(stof(cell));
            }

            vector_data.push_back(train_data_vector_2);
        }
    }

    cout << "\tfinished loading" << endl;

    return vector_data;
}

void ConvertVectortoMatrix(vector<vector<float>> &In_vector, Mat &Out_Mat)
{

	int descriptor_size = In_vector[0].size();

	for (int i = 0; i < In_vector.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			Out_Mat.at<float>(i, j) = In_vector[i][j];
		}
	}

}
