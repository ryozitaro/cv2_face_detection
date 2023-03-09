import cv2
import numpy as np
from box_utils import predict

ONNX_FILE = cv2.dnn.readNetFromONNX("version-RFB-640.onnx")


def detection(image):
    array = np.frombuffer(image.getvalue(), np.int8)
    orig_image = cv2.imdecode(array, 1)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(orig_image, (640, 480))

    # 推論前処理
    image = preprocess(image)

    # 顔の位置を特定
    scores, boxes = get_face_position(image)

    # 推論後処理
    boxes, labels, probs = predict(
        orig_image.shape[1], orig_image.shape[0], scores, boxes, 0.7
    )

    # ボックスを描画
    for box in boxes:
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)

    return orig_image


def get_face_position(image):
    onnx = ONNX_FILE
    onnx.setInput(image)
    input_name = onnx.getLayerNames()[-2:]
    scores, boxes = onnx.forward(input_name)
    return scores, boxes


def preprocess(image):
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, 0)
    image = image.astype(np.float32)
    return image
