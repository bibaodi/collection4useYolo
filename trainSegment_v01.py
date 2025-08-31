import pathlib
from ultralytics import YOLO
#- eton@250831 add eval, and test image;

# Load a pretrained YOLO11 segment model
model = YOLO("yolo11n-seg.pt")

# Train the model
#results = model.train(data="coco8-seg.yaml", epochs=1, imgsz=640)
#results = model.train(data="thyNoduSeg.yaml", epochs=1, imgsz=640)
#results = model.train(data="../datasets/segThyGland_v02.yaml", epochs=30, imgsz=224)

results = model.train(data="../datasets/segThyNodunGland_v01.yaml", epochs=3, imgsz=224)
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered

# Export the model to ONNX format
success=42
#success = model.export(format="onnx")
# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")
tobeTestImg=r'../datasets/42-minibatch/thynodu-t03.jpg'
testImg=pathlib.Path(tobeTestImg)
if testImg.is_file():
    results = model.predict(tobeTestImg)
else:
    print(f"test img not exist:{testImg}")
print(f"export to ONNX: success={success}")

exportONNX=0
if exportONNX==1:
# Export the model to ONNX format
    success = model.export(format="onnx")
    print(f"export to ONNX: success={success}")
