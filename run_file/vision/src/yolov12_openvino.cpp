//
// Created by yu on 2025/4/1.
//
#include "yolov12_openvino.hpp"

using namespace cv;
using namespace ov;

void MODEL::ImgPreprocess(Mat &inputImg, int newSize)
{
    // 读入图片变换尺寸
    imgSize = newSize;
    // 图片尺寸变换并写入共享指针
    resize(inputImg, *img, Size(newSize, newSize), 0, 0, INTER_LINEAR);
}

void MODEL::OpenVinoInitial(const string& xmlPath)
{
    Core core;
    shared_ptr<Model> openvinoModel = core.read_model(xmlPath);
    // 初始化模型的预处理
    preprocess::PrePostProcessor ppp = preprocess::PrePostProcessor(model);
    // 指定输入图像格式
    ppp.input().tensor().set_element_type(element::u8).set_layout("NHWC").set_color_format(preprocess::ColorFormat::BGR);
    // 指定输入图像的预处理管道而不调整大小
    ppp.input().preprocess().convert_element_type(element::f32).convert_color(preprocess::ColorFormat::RGB).scale({ 255., 255., 255. });
    // 指定模型的输入布局
    ppp.input().model().set_layout("NCHW");
    // 指定输出结果格式
    ppp.output().tensor().set_element_type(element::f32);
    // 在图形中嵌入以上步骤
    openvinoModel = ppp.build();
    compiled_model = core.compile_model(model, "CPU");

}

shared_ptr<MODEL::InferenceResult> MODEL::Inference()
{
    return shared_ptr<InferenceResult>();
}
