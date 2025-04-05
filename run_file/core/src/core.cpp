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
using namespace std;

int main()
{
    // 读配置参数
    _config.ReadConfigParameters();
    // openvino初始化
    CompiledModel compiledModel;
    MODEL::OpenVinoInitial(_config._parameters.xml_path, compiledModel);
    // 初始化模型设置

    string path = "/home/yu/文档/erlangShen_ws/yolov12/3.jpg";
    Mat img = imread(path);
    model.ImgPreprocess(img, 640);

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
    auto out_date = model.Inference(compiledModel);
    model.Postprocess(out_date);
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;


    std::vector<std::string> class_names = {"0", "1"};
    model.DrawResults(*model.img, class_names);

    printf("时间：%f", duration.count());
}