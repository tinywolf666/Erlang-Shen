import openvino as ov

model = ov.convert_model(
    input_model='/home/yu/文档/erlangShen_ws/yolov12/best-320-sim.onnx'
)

# 可选：保存为 IR 格式
ov.save_model(model, 'best-320.xml')
