from roboflow import Roboflow

rf = Roboflow(api_key="X3R2wlGX5wr1dP7dHnbR")
workspace = rf.workspace('erlangshen')

workspace.deploy_model(
    model_type="yolov12n",
    model_path="/home/yu/文档/erlangShen_ws/yolov12/models",
    project_ids=["target-recognition"],
    model_name="erlangShen-640"
)