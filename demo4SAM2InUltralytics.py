import os
from ultralytics import SAM
from ultralytics import YOLO
from ultralytics import settings

# View all settings
print(settings)

def useYolo11Seg_1(tobeTestImg):

    # Load a pretrained model
    model = YOLO("yolo11n-seg.pt")
    # Validate the model
    #metrics = model.val()
    #print("Mean Average Precision for boxes:", metrics.box.map)
    #print("Mean Average Precision for masks:", metrics.seg.map)

    # Display model information (optional)
    model.info()

    # Run inference with bboxes prompt
    print(f"test image:[{tobeTestImg}]")

    results = model(tobeTestImg)
    #print(results)
    for r in results:
        print(f"Detected {len(r)} objects in image")   
        # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
 
def useSAM2_1(tobeTestImg):
    # Load a model
    model = SAM("sam2.1_b.pt")

    # Display model information (optional)
    model.info()

    print(f"test image:[{tobeTestImg}]")


    # Run inference with single point
    results = model(tobeTestImg, points=[501, 195], labels=[1])

    #print(results)
    for r in results:
        print(f"Detected {len(r)} objects in image")   
        r.show()  # display to screen
        
    return 0
    # Run inference with bboxes prompt
    results = model(tobeTestImg, bboxes=[452, 141, 549, 250])

    # Run inference with multiple points
    results = model(points=[[473, 183], [514, 226]], labels=[1, 1])

    # Run inference with multiple points prompt per object
    results = model(points=[[[473, 183], [514, 226]]], labels=[[1, 1]])

    # Run inference with negative points prompt
    results = model(points=[[[501, 195], [523, 265]]], labels=[[1, 0]])

def runSegmentModelPredict():
    imgHome=r'/home/eton/00-src/250101-YOLO-ultralytics/'
    tobeTestImg=imgHome+r'datasets/yoloDataset01.V0/images/val/301PACS02-2401010320_frm-0001.png'
    useSAM2_1(tobeTestImg)
    #useYolo11Seg_1(tobeTestImg)

if __name__ == "__main__":
    runSegmentModelPredict()

