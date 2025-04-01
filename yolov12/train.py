from ultralytics import YOLO

model = YOLO("best.pt")

# results = model.train(
#     data='dataset/data.yaml',
#     epochs=200,
#     batch=30,
#     workers=24,
#     imgsz=640,
#     lr0=0.0001,
#     scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
#     mosaic=1.0,
#     mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
#     copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
#     optimizer="Adam",
#     device="0",
# )
#
# metrics = model.val()

# Perform object detection on an image
results = model("3.jpg")
results[0].show()