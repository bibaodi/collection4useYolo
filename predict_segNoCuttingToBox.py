# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as torchnnF

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS

def crop_mask(masks, boxes):
    """
    Crop masks to bounding boxes.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks.
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form.

    Returns:
        (torch.Tensor): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    print(f"debug: masks.shape={masks.shape}, boxes.shape={boxes.shape}, boxes={boxes}, x1={x1}, y1={y1}, x2={x2}, y2={y2}\n w={w}, h={h}")
    boxW= x2 - x1
    boxH= y2 - y1
    x1 = x1 - boxW // 2
    y1 = y1 - boxH // 2
    x2 = x2 + boxW // 2
    y2 = y2 + boxH // 2
    x1 = torch.clamp(x1, 0, w)
    y1 = torch.clamp(y1, 0, h)
    x2 = torch.clamp(x2, 0, w)
    y2 = torch.clamp(y2, 0, h)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    #masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = torchnnF.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.0)

class YOLOSeg:
    """
    YOLOv8 segmentation model for performing instance segmentation using ONNX Runtime.

    This class implements a YOLOv8 instance segmentation model using ONNX Runtime for inference. It handles
    preprocessing of input images, running inference with the ONNX model, and postprocessing the results to
    generate bounding boxes and segmentation masks.

    Attributes:
        session (ort.InferenceSession): ONNX Runtime inference session for model execution.
        imgsz (Tuple[int, int]): Input image size as (height, width) for the model.
        classes (dict): Dictionary mapping class indices to class names from the dataset.
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold used by non-maximum suppression.

    Examples:
        >>> model = YOLOv8Seg("yolov8n-seg.onnx", conf=0.25, iou=0.7)
        >>> img = cv2.imread("image.jpg")
        >>> results = model(img)
        >>> cv2.imshow("Segmentation", results[0].plot())
    """

    def __init__(self, onnx_model, conf=0.25, iou=0.7, imgsz=640):
        """
        Initialize the instance segmentation model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            imgsz (int | Tuple[int, int]): Input image size of the model. Can be an integer for square input or a tuple
                for rectangular input.
        """
        self.argtask = "segment"
        print("debug onnx provider:", ort.__version__, ort.get_device() )
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"],
        )

        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.conf = conf
        self.iou = iou

    def __call__(self, img):
        """
        Run inference on the input image using the ONNX model.

        Args:
            img (np.ndarray): The original input image in BGR format.

        Returns:
            (List[Results]): Processed detection results after post-processing, containing bounding boxes and
                segmentation masks.
        """
        prep_img = self.preprocess(img, self.imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs)

    def letterbox(self, img, new_shape=(640, 640)):
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image in BGR format.
            new_shape (Tuple[int, int]): Target shape as (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img

    def preprocess(self, img, new_shape):
        """
        Preprocess the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (Tuple[int, int]): The target shape for resizing as (height, width).

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized.
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255  # Normalize to [0, 1]
        return img

    def postprocess(self, img, prep_img, outs):
        """
        Post-process model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (list): Model outputs containing predictions and prototype masks.

        Returns:
            (List[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        
        preds, protos = [torch.from_numpy(p) for p in outs]
        print(f"debug: {type(preds)}preds.shape={preds.shape}, protos.shape={protos.shape}")
        model_names={0: 'thyNodu'} 
        preds = ops.non_max_suppression(preds, self.conf, self.iou, None, False, max_det=3000, nc=len(model_names), end2end=False, return_idxs=False)

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            #masks = process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names={'nodu':"Nodule"}, boxes=pred[:, :6], masks=masks), )

        return results

    def process_mask(self, protos, masks_in, bboxes, shape):
        """
        Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

        Args:
            protos (torch.Tensor): Prototype masks with shape (mask_dim, mask_h, mask_w).
            masks_in (torch.Tensor): Predicted mask coefficients with shape (n, mask_dim), where n is number of detections.
            bboxes (torch.Tensor): Bounding boxes with shape (n, 4), where n is number of detections.
            shape (Tuple[int, int]): The size of the input image as (height, width).

        Returns:
            (torch.Tensor): Binary segmentation masks with shape (n, height, width).
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # Matrix multiplication
        masks = ops.scale_masks(masks[None], shape)[0]  # Scale masks to original image size
        masks = crop_mask(masks, bboxes)  # Crop masks to bounding boxes
        return masks.gt_(0.0)  # Convert to binary masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.1, help="NMS IoU threshold")
    args = parser.parse_args()

    model = YOLOSeg(args.model, args.conf, args.iou)
    img = cv2.imread(args.source)
    results = model(img)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    #result.save(filename="/tmp/result-nocut.jpg")  # save to disk

    img = np.copy(result.orig_img)
    b_mask = np.zeros(img.shape[:2], np.uint8)
    print(f"debug: {type(masks.xy)}, len={len(masks.xy)}, masks={masks}") if len(masks.xy) <= 0 else print("a lot masks found")
    contour = masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    isolated = np.dstack([img, b_mask])
    cv2.imwrite("/tmp/resnocut-isolated.png", isolated)
    lineThickness=1
    cv2.drawContours(img, [contour], -1, (55, 255, 25),thickness=lineThickness,lineType=cv2.LINE_AA )#color:BGR, 
    cv2.imwrite("/tmp/resnocut-contour.png", img)
