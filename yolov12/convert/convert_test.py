from ultralytics import YOLO

model = YOLO('/home/yu/文档/erlangShen_ws/yolov12/best-640.pt')
model.export(format="openvino", half=True)  # or format="onnx"