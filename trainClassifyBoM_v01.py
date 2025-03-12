from ultralytics import YOLO
import pathlib


# - eton@250215 change dataset home from absolute to relative;

# Create a new YOLO model from scratch
#model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11-cls-resnet18.yaml")  # load modeli yolo11-cls-resnet18.yaml from yaml 
#model = YOLO("yolo11m-cls.pt")  # load maxs pretrained model

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# datasetf=r'../datasets/301Pacs978Items4ObjectDetectBenignMalignV1_250109/301Pacs978Items4ObjectDetectBenignMalign.yaml'
datasetHome=r'../datasets/'
datasetName=r"301pacsDataInLbmfmtRangeY22-24.clsBoM_extend2times"
datasetName=r"clsBoM_v01_extend2times"
datasetName=r"clsBoM_v03test"
trainMessage="2cd train to classify Benign Malign model, 8/2 for train/val, delete too small images"
trainMessage="3rd test on dong5k images only"

datasetf=datasetHome+datasetName

# Train the model
#results = model.train(data="mnist", epochs=10, imgsz=32)
results = model.train(data=datasetf, epochs=1, imgsz=96, erasing=0.1)#96

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1  # top1 accuracy

# Export the model to ONNX format
success = model.export(format="onnx")
# Perform object detection on an image using the model
#results = model("https://ultralytics.com/images/bus.jpg")
tobeTestImg=datasetf+r'/test/malign/301PACS02-2201041312.01_frm-0002.png'
testImg=pathlib.Path(tobeTestImg)
if testImg.is_file():
    results = model.predict(tobeTestImg)
else:
    print(f"test img not exist:{testImg}")
print(f"export to ONNX: success={success}")

