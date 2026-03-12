from ultralytics import YOLO
import numpy as np
import camera_capture as cam_cap


class YoloDetect:
    """Class to handle YOLO object detection on a separate thread.
    """
    def __init__(self, model_path='YOLO/runs/yolo-familiar-faces9/weights/best.onnx'):
        
        self.model = YOLO(model_path, task='detect')

    def predict(self, frame):

        results = self.model(frame, verbose=False)
        for result in results:
            return result.boxes.xywh