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
    Mat sol_img = inputImg.clone();
    int cols = sol_img.cols;
    int rows = sol_img.rows;

    // 先将图片裁减为正方形
    if (cols >= rows)
    {
        sol_img = sol_img(Range(0, rows),
                          Range((int) (cols - rows) / 2, (int) (rows + (cols - rows) / 2)));
    } else
    {
        sol_img = sol_img(Range((int) (rows - cols) / 2, (int) (cols + (rows - cols) / 2)),
                          Range(0, cols));
    }


    // 图片尺寸变换并写入共享指针
    resize(sol_img, *img, Size(newSize, newSize), 0, 0, INTER_LINEAR);
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

const float *MODEL::Inference(CompiledModel &compiledModel)
{
    auto *input_data = (float *) img->data;

    Tensor input_tensor(compiledModel.input().get_element_type(), compiledModel.input().get_shape(), input_data);
    InferRequest infer_request = compiledModel.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    const Tensor &output_tensor = infer_request.get_output_tensor();
    const float *output_data = output_tensor.data<float>();

    return output_data;
}

// 后处理
void MODEL::Postprocess(const float *output_data)
{
    // 解析输出数据
    // 1. 解析输出数据
    // 2. 非极大值抑制
    // 3. 阈值过滤
    // 4. 返回结果

    // 解析输出数据
    int num_pres = 8400;
    std::vector<InferenceResult> temp_results;

    for (int i = 0; i < num_pres; ++i)
    {
        float x = output_data[0 * num_pres + i];
        float y = output_data[1 * num_pres + i];
        float w = output_data[2 * num_pres + i];
        float h = output_data[3 * num_pres + i];
        float conf = output_data[4 * num_pres + i];
        float cls_id = output_data[5 * num_pres + i];

        InferenceResult r;
        r.box = Rect2f(x, y, w, h);
        r.confidence = conf;
        r.class_id = (cls_id > 0.5f) ? 1 : 0;

        if (r.class_id == 1 && conf > 0.00001f)
        {
            r.confidence *= 10000;  // 你之前的特殊处理
            temp_results.push_back(r);
        } else if (r.class_id == 0 && conf > 0.01f)
        {
            temp_results.push_back(r);
        }
    }
    // 非极大值抑制

    float iou_thresh = 0.3f;
    float score_thresh = 0.01f;

    for (int cls = 0; cls <= 1; ++cls)
    {
        vector<Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> indices;
        std::vector<InferenceResult> class_results;

        for (const auto &r: temp_results)
        {
            if (r.class_id == cls)
            {
                boxes.push_back(r.box);
                scores.push_back(r.confidence);
                class_results.push_back(r);
            }
        }

        dnn::NMSBoxes(boxes, scores, score_thresh, iou_thresh, indices);

        for (int idx: indices)
        {
            results->push_back(class_results[idx]);
        }
    }


    for (const auto &item: *results)
    {
        printf("class: %d  confidence: %.2f\n", item.class_id, item.confidence);
    }
}

void MODEL::DrawResults(Mat &image, const std::vector<std::string> &class_names)
{
    for (const auto &r: *results)
    {

        Scalar color(0, 255, 0);
        if (r.class_id == 1)
        {
            color = Scalar(255, 0, 0);
            Rect center_rect;
            center_rect = Rect(r.box.x - r.box.width / 2, r.box.y - r.box.height / 2,
                               r.box.width, r.box.height);
            rectangle(image, center_rect, color, 1);
            // 组装标签文字：类别名 + 置信度
            std::string label = "Target";

            // 文字尺寸 & 背景框
            int baseLine;
            Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            Point label_origin(center_rect.x, center_rect.y - 5);
            rectangle(image, label_origin + Point(0, baseLine),
                      label_origin + Point(label_size.width, -label_size.height),
                      Scalar(255,0,255), FILLED);

            // 写文字
            putText(image, label, label_origin, FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(0, 0, 0), 1);

            // 画靶心
            Point targetCenter(r.box.x, r.box.y);
            // circle(image, targetCenter, 3, Scalar(0, 0, 255), -1);
        } else if (r.class_id == 0)
        {
            auto radius = min(r.box.height, r.box.width) / 2;
            ellipse(image, Point(r.box.x, r.box.y), Size(radius, radius), 0, 0, 360, Scalar(0, 255, 0),
                    1);
        }
    }
    // imwrite("/home/yu/文档/erlangShen_ws/out3.jpg", image);
    namedWindow("Image", WINDOW_GUI_NORMAL);
    imshow("Image", image);
    waitKey(-1);
}
