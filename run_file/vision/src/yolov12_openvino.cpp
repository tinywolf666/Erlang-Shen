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
    // 清空上一帧的推理结果
    results->clear();
}

void MODEL::OpenVinoInitial(const string &xmlPath, CompiledModel &compiled_model)
{
    Core core;
    shared_ptr<Model> openvinoModel = core.read_model(xmlPath);

    // 初始化模型的预处理
    preprocess::PrePostProcessor ppp = preprocess::PrePostProcessor(openvinoModel);
    // 指定输入图像格式
    ppp.input().tensor().set_element_type(element::u8).set_layout("NHWC").set_color_format(
            preprocess::ColorFormat::BGR);
    // 指定输入图像的预处理管道而不调整大小
    ppp.input().preprocess().convert_element_type(element::f32).convert_color(preprocess::ColorFormat::RGB).scale(
            {255., 255., 255.});
    // 指定模型的输入布局
    ppp.input().model().set_layout("NCHW");
    // 指定输出结果格式
    ppp.output().tensor().set_element_type(element::f32);
    // 在图形中嵌入以上步骤
    openvinoModel = ppp.build();
    compiled_model = core.compile_model(openvinoModel, "CPU");

}

void MODEL::Inference(CompiledModel &compiledModel)
{
    auto *input_data = (float *) img->data;

    Tensor input_tensor(compiledModel.input().get_element_type(), compiledModel.input().get_shape(), input_data);
    InferRequest infer_request = compiledModel.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    const Tensor &output_tensor = infer_request.get_output_tensor();
    const float *output_data = output_tensor.data<float>();
    Shape shape = output_tensor.get_shape();

    for (int i = 0; i < shape[1]; ++i)
    {
        const float *det = &output_data[i * shape[2]];

        InferenceResult result;
        result.box = Rect2f(det[0], det[1], det[2], det[3]);  // raw bbox values
        result.confidence = det[4];
        result.class_id = max_element(det + 5, det + shape[2]) - (det + 5);

        results->push_back(result);
    }

    for (const auto &item: *results)
    {
        printf("class: %d  confidence: %.2f\n", item.class_id, item.confidence);
    }


}
