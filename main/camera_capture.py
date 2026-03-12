
import math
import cv2
import numpy as np
import threading
import yolo_detect as yolo_det

# Class was generated with AI. All functions were read through and understood by the author.
# The code follows the design standard desird for this project. Multithreadding is a challenging topic, 
# which is why AI was used to to assist in the development of this class.
class CameraCapture:
    """Create a camera frame stream on a dedicated thread.
    """
    def __init__(self, camera_index=0):
        """Initialize the CameraCapture class

        Args:
            camera_index (int, optional): Select which hardware camera to use. Defaults to 0.
        """
        self.camera_index = camera_index
        self.capture = None
        self.frame = None
        self.running = False
        self.thread = None

    def start(self):
        """Start the camera capture loop

        Raises:
            Exception: Raise error if camera cannot be opened
        """
        if not self.running:
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                raise Exception("Could not open camera")
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.start()

    def _capture_loop(self):
        """Capture frames from the camera and pass them to the `self.frame` variable for use in the main thread.
        """
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame

    def stop(self):
        """End the camera process and empty the thread.
        """
        if self.running:
            self.running = False
            self.thread.join()
            self.capture.release()

    def get_frame(self):
        """Get the most recent frame from the video stream.

        Returns:
            Tuple: Camera frame.
        """
        return self.frame
    

class KalmanFilter():
    #https://pieriantraining.com/kalman-filter-opencv-python-example/

    #https://www.bacancytechnology.com/qanda/python/opencv-kalman-filter-with-python
    def __init__(self, id, hsv_frame, track_window):
        self.id = id
        self.track_window = track_window
        self.decay = 0

        x ,y, w, h = self.track_window
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 0.95, 0],
             [0, 0, 0, 0.95]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.0001
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        cx = x+w/2
        cy = y+h/2

        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0],  [0]], np.float32)

    def predict(self):
        self.decay += 1
        prediction = self.kalman.predict()
        return prediction
    
    def update(self, track_window):
        self.decay = 0
        x ,y, w, h = track_window
        cx = x+w/2
        cy = y+h/2
        measurement = np.array([[cx], [cy]], np.float32)
        self.kalman.correct(measurement)

    def get_position(self):
        # Generated with the help of Gemini. I have reviewed the code and understand
        # how it works. I am new to using Kalman filters, so I have used AI to help me
        # understand what functions I need to use and how to use them.
        # 1. Get the predicted state
        prediction = self.kalman.predict()

        # 2. Extract center x and center y (the first two values)
        pred_cx = prediction[0][0]
        pred_cy = prediction[1][0]

        # 3. Calculate the top-left corner
        pred_x = int(pred_cx - (self.w / 2))
        pred_y = int(pred_cy - (self.h / 2))

        # Return the top-left x, y, and the width, height
        return pred_x, pred_y, self.w, self.h

if __name__ == "__main__":
    camera = CameraCapture()
    camera.start()
    yolo = yolo_det.YoloDetect()
    frame_num = 0
    id = 0
    active_filters = []

    try:
        while True:
            # Run the camera frame capture loop
            frame = camera.get_frame()

            if frame is not None:
                frame_num += 1

                # Create initial predictions
                for kf in active_filters:
                    kf.predict()
                # Pass frame to YOLO model return face_frame
                if frame_num % 10 == 0 or frame_num == 1:
                    track_window = yolo.predict(frame)

                    # If the YOLO model is detecting a face, call the Kalman filter class
                    if track_window.shape[0] > 0:
                        # For every box detected by YOLO, search to see if there is a Kalman filter nearby
                        for window in track_window:
                            x, y, w, h = window
                            best_match = None
                            nearest_match = 200

                            # For every filter, search for the closest one to the center of the detected box.
                            # If there is a match, update the filter. Otherwise, create a new filter.
                            for kf in active_filters:
                                pred_cx, pred_cy, _, _ = kf.get_position()
                                distance = math.hypot(x - pred_cx, y - pred_cy)
                                if distance < nearest_match:
                                    nearest_match = distance
                                    best_match = kf
                                # If the Kalamn filter has not been updated for 5 frames, remove it from the active filters list
                                if kf.decay > 15:
                                    active_filters.remove(kf)
                            if best_match is not None:
                                best_match.update(window)
                            else:
                                new_filter = KalmanFilter(id, frame, window)
                                active_filters.append(new_filter)
                                id += 1

                for kf in active_filters:
                    x, y, w, h = kf.get_position()
                    cv2.rectangle(frame, (int(x-(w//2)), int(y-(h//2))), (int(x + (w//2)), int(y + (h//2))), (0, 255, 0), 2)

                # Display the frame
                cv2.imshow("Familiar Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

