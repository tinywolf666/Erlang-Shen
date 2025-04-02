#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class MODEL
{
private:
    int imgSize = 1;
    shared_ptr<Mat> img = make_shared<Mat>(Mat::zeros(imgSize, imgSize, CV_8UC3));
public:
//    推理结果结构体
    struct InferenceResult
    {
        int class_id;
        float confidence;
        Rect bbox;
    };

//  图片预处理
    void ImgPreprocess(Mat &inputImg, int newSize);

//  openvino初始化
    void OpenVinoInitial(const string& xmlPath);

//  推理
    shared_ptr<InferenceResult> Inference();
} inline model;