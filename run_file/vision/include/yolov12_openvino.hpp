#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
//  推理结果结构体
struct InferenceResult
{
    int class_id;
    float confidence;
    Rect box;
};

class MODEL
{
private:
    int imgSize = 1;
    shared_ptr<Mat> img = make_shared<Mat>(Mat::zeros(imgSize, imgSize, CV_8UC3));
    shared_ptr<vector<InferenceResult>> results = make_shared<vector<InferenceResult>>();
public:

//  图片预处理
    void ImgPreprocess(Mat &inputImg, int newSize);

//  openvino初始化
    static void OpenVinoInitial(const string &xmlPath, ov::CompiledModel &compiled_model);

//  推理
    void Inference(ov::CompiledModel &compiledModel);
} inline model;