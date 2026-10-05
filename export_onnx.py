from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=320,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=12,
    nms=False
)
