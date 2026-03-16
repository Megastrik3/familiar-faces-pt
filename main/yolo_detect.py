from ultralytics import YOLO


class YoloDetect:
    """Class to handle YOLO object detection on a separate thread.
    """
    def __init__(self, model_path='YOLO/runs/yolo-familiar-faces9/weights/best.onnx'):
        """Perform object detection on image frame using YOLOv26 mode..

        Args:
            model_path (str, optional): Path to YOLOv26-Face model. Defaults to 'YOLO/runs/yolo-familiar-faces9/weights/best.onnx'.
        """
        self.model = YOLO(model_path, task='detect')

    def predict(self, frame):
        """Detect faces in frame using YOLO model.

        Args:
            frame (array): The input frame for face detection.

        Returns:
            _type_: The bounding boxes of detected faces.
        """
        results = self.model(frame, verbose=False)
        for result in results:
            return result.boxes.xywh