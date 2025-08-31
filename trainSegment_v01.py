from ultralytics import YOLO

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n-seg.pt")

# Train the model
#results = model.train(data="coco8-seg.yaml", epochs=1, imgsz=640)
#results = model.train(data="thyNoduSeg.yaml", epochs=1, imgsz=640)
results = model.train(data="../datasets/segThyGland_v02.yaml", epochs=30, imgsz=224)
# Export the model to ONNX format
success = model.export(format="onnx")
print(f"export to ONNX: success={success}")

