from ultralytics import YOLO

model = YOLO('/home/yu/yolov12/runs/detect/train/weights/best.pt')
model.export(format="openvino", half=True, simplify=True, imgsz=640)  # or format="onnx"
