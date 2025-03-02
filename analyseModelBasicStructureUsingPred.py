#!/bin/python
import cv2
import numpy as np
import sys
from PIL import Image
from ultralytics import YOLO

import onnxruntime as onnx

s_score_threshold=0

def process_mode_output(pred):
    predictions = pred[0]
    print(f"prediction shape={predictions.shape}")
    predictions = predictions.transpose((1, 0))
    print(f"prediction shape={predictions.shape}")

    scores = predictions[:, 4]
    print(f"prediction score={scores}")
    high_conf_mask = scores > s_score_threshold
    high_conf_preds = predictions[high_conf_mask]

    if len(high_conf_preds) == 0:
        max_conf_idx = scores.argmax()
        high_conf_preds = predictions[max_conf_idx:max_conf_idx + 1]
        return [[x - w / 2, y - h / 2, x + w / 2, y + h / 2, conf]
                for x, y, w, h, conf in high_conf_preds]

    sorted_indices = np.argsort(-high_conf_preds[:, 4])
    high_conf_preds = high_conf_preds[sorted_indices]

    max_conf_box = high_conf_preds[0]
    selected_boxes = [max_conf_box]

def predict_with_onnx(modelfile, imagefile):
    model_sess = onnx.InferenceSession(modelfile)
    inputs=model_sess.get_inputs()
    outputs=model_sess.get_outputs()

    print(f"onnx {'='*64}inputs-type={type(inputs)}")
    print(f"model: {model_sess}")
    for input in inputs:
        print(f"input-type={type(input)}, {input}, dir(input)")
    #print(f"outputs-type={type(outputs)}")
    for output in outputs:
        print(f"output-type={type(output)}, {output}")
#input-type=<class 'onnxruntime.capi.onnxruntime_pybind11_state.NodeArg'>,
    #NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
#output-type=<class 'onnxruntime.capi.onnxruntime_pybind11_state.NodeArg'>, 
    #NodeArg(name='output0', type='tensor(float)', shape=[1, 5, 8400])
    inp=inputs[0]
    inp_name=inp.name
    inp_shape = inp.shape #[-2:]

    oup=outputs[0]
    oup_name=oup.name
    imgData=cv2.imread(imagefile)
    imgData.resize(inp_shape)
    imgData = imgData.astype(np.float32) #/ 255.0
    #image actual size=(768, 1024, 3)
    print(f"image actual size={imgData.shape}, modelInputShape={inp_shape}")
    results = model_sess.run([oup_name], {inp_name: imgData})
    NofRets=len(results)
    print(f"results type={type(results)}, len={len(results)}")
    if NofRets>0:
        ret0=results[0]
        process_mode_output(ret0)
        print(f"result type={type(ret0)}, shape={ret0.shape}")
        predNd=ret0[0]#5,8400;
        # Condition: M0[4, :] > 0.1
        condition = predNd[4, :] > 0.1
        # Extract the submatrix using the condition
        subPred = predNd[:, condition]
        
        for irow in range( max(subPred.shape[1], 5)):
            ipred=subPred[:, irow]
            print(f"{irow}: shape={ipred.shape}, top10={ipred.tolist()}")
        #print(f"result={ret0}") 
        #print(f"boxes:{ret0.boxes}")
    return

def predict_TIRADS_01(modelfile, imagefile):
    model = YOLO(modelfile)
    # Run inference on 'bus.jpg'
    #results = model([imagefile])  # results list
    results = model.predict([imagefile])  # results list
    NofRets=len(results)
    print(f".pt {'='*64}results type={type(results)}, len={len(results)}")
    if NofRets>0:
        ret0=results[0]
        print(f"result type={type(ret0)}")
        #print(f"result={ret0}")
        
        print(f"boxes:{ret0.boxes}")

    return 
    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Show results to screen (in supported environments)
        r.show()

        # Save results to disk
        #r.save(filename=f"results{i}.jpg")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print(f"Usage: {sys.argv[0]} <model> <image>")
        sys.exit(1)
    m=sys.argv[1]
    m=r'/home/eton/00-src/250101-YOLO-ultralytics/42-250107-train_BenignMalign/runs/detect/train16/weights/best.pt'
    mx=m.replace('.pt', '.onnx')
    img=sys.argv[2]
    img=r'/home/eton/00-src/250101-YOLO-ultralytics/datasets/yoloDataset01/images/train/301PACS02-2401010411.01_frm-0002.png'
    img=r'687.jpg'
    #img=sys.argv[2]
    predict_TIRADS_01(m, img)
    predict_with_onnx(mx, img)
#
