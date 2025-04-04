//
// Created by yu on 2025/4/1.
//
#include <openvino/openvino.hpp>
#include <chrono>

#include "yolov12_openvino.hpp"
#include "config.hpp"
#include "core.hpp"

using namespace ov;
using namespace cv;


int main()
{
    // 读配置参数
    _config.ReadConfigParameters();
    // openvino初始化
    CompiledModel compiledModel;
    MODEL::OpenVinoInitial(_config._parameters.xml_path, compiledModel);
    // 初始化模型设置

    string path = "/home/yu/文档/erlangShen_ws/yolov12/1.jpg";
    Mat img = imread(path);
    model.ImgPreprocess(img, 640);
    auto start = std::chrono::high_resolution_clock::now();
    model.Inference(compiledModel);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("时间：%f", duration.count());
}