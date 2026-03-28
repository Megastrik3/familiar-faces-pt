from ultralytics import YOLO
import threading

## The following class was originally written by hand. However, I was encoutering some threading
# issues which required a change to the structure of the code. I used AI to help me understand how
# I should use threading to run the YOLO model in the background. I used the suggestions from the AI
# to rewrite the class. I read through the code and understand how it works and have modified it as needed.
# https://gemini.google.com/share/f99adf681b19

class YoloDetect:
    """Class to handle YOLO object detection on a separate thread.
    """
    def __init__(self, model_path='YOLO/runs/yolo-familiar-faces9/weights/best.onnx'):
        """Perform object detection on image frame using YOLOv26 mode..

        Args:
            model_path (str, optional): Path to YOLOv26-Face model. Defaults to 'YOLO/runs/yolo-familiar-faces9/weights/best.onnx'.
        """
        self.model = YOLO(model_path, task='detect')
        self.latest_boxes = []
        self.new_boxes = False
        self.running = False
        self.thread = None
        self.frame_skip = 10
    
    def start(self, camera):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._detect_loop, args=(camera,))
            self.thread.start()

    def _detect_loop(self, camera):
        frame_count = 0
        while self.running:
            frame = camera.get_frame()
            if frame is not None:
                frame_count += 1
                boxes = []
                if frame_count % self.frame_skip == 0:
                    results = self.model(frame, verbose=False)
                    for result in results:
                        boxes = result.boxes.xywh.cpu().numpy() if len(result.boxes) > 0 else []
                    frame_count = 0
                    
                    self.latest_boxes = boxes
                    self.new_boxes = True
        

    def get_latest_boxes(self):
        """Get the most recent bounding boxes from the YOLO model.

        Returns:
            List: List of bounding boxes in the format [x_center, y_center, width, height].
        """
        self.new_boxes = False
        return self.latest_boxes
    
    def stop(self):
        """Stop the YOLO detection thread.
        """
        if self.running:
            self.running = False
            self.thread.join()