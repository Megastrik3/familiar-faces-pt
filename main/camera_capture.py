
import math
import time
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
        self.yolo = yolo_det.YoloDetect()

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
        
    def predictFrame(self):
        return self.yolo.predict(self.frame)


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
    

# I have never worked with Kalman filters before, so I used a variety of tools to help me understand how to use them,
# and how to implement them.
# I used the following two articles to give me a basic understanding of how to structure my Kalman filter class
# and what types of operations and initializations I would need to use:
    #https://pieriantraining.com/kalman-filter-opencv-python-example/
    #https://www.bacancytechnology.com/qanda/python/opencv-kalman-filter-with-python
# Then, I used Google Gemini to help me understand how to use that knowledge for my particular application.
# While Gemini helped me get started, I quickly understood what the various operations and functions were doing,
# and was able to customize them to my application. I have throughly reviewed the code and understand how it is
# working for my application. I have included the chat history for the Kalman filter below:
    #https://gemini.google.com/share/4bc3e63296c2
class KalmanFilter():
    """Class to implement a Kalman filter
    """
    def __init__(self, id, track_window):
        """Create a Kalman filter to track faces throughout the frame.

        Args:
            id (int): Fitler identifier, used to keep track of multiple filters
            track_window (array): The coordienates of the box detected by the YOLO model, used to initialize the Kalman filter
        """
        self.id = id
        self.track_window = track_window
        self.decay = 0
        self.prediction = None

        x ,y, w, h = self.track_window
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.dt = 0.5

        self.kalman = cv2.KalmanFilter(6, 2, 0)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, self.dt, 0, 0.5*self.dt**2, 0],
             [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
             [0, 0, 0.95, 0, self.dt, 0],
             [0, 0, 0, 0.95, 0, self.dt],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        cx = x+w/2
        cy = y+h/2

        self.kalman.statePre = np.array([[cx], [cy], [0], [0], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0],  [0], [0], [0]], np.float32)

    def predict(self, fps=None):
        """Generate bounding box prediction from initial frame.

        Returns:
            Prediction: Returns the predicted bounding box coordinates.
        """
        if fps is not None:
            self.dt = 1/fps
            self.kalman.transitionMatrix = np.array(
            [[1, 0, self.dt, 0, 0.5*self.dt**2, 0],
             [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
             [0, 0, 0.95, 0, self.dt, 0],
             [0, 0, 0, 0.95, 0, self.dt],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]], np.float32)
        self.decay += 1
        self.prediction = self.kalman.predict()
        return self.prediction
    
    def update(self, track_window):
        """Update the current location of the predicted bounding box.

        Args:
            track_window (array): The coordinates of the box detected by the YOLO model, used to update the Kalman filter
        """
        self.decay = 0
        x ,y, w, h = track_window
        cx = x+w/2
        cy = y+h/2
        measurement = np.array([[cx], [cy]], np.float32)
        self.kalman.correct(measurement)

    def get_position(self):
        """Get the position of the predicted bounding box.

        Returns:
            Tuple: The coordinates of the predicted bounding box.
        """
        self.prediction = self.kalman.predict()
        pred_cx = self.prediction[0][0]
        pred_cy = self.prediction[1][0]

        pred_x = int(pred_cx - (self.w / 2))
        pred_y = int(pred_cy - (self.h / 2))

        return pred_x, pred_y, self.w, self.h

    def get_acceleration(self):
        """Get the acceleration of a prediction

        Returns:
            Tuple[float, float]: The acceleration values (acc_x, acc_y).
        """
        acc_x = self.prediction[4][0]
        acc_y = self.prediction[5][0]
        return acc_x, acc_y


if __name__ == "__main__":
    camera = CameraCapture()
    camera.start()
    total_frames = 0
    frame_num = 0
    id = 0
    active_filters = []

    try:
        while True:
            fps = camera.capture.get(cv2.CAP_PROP_FPS)
            # Run the camera frame capture loop
            frame = camera.get_frame()

            if frame is not None:
                frame_num += 1

                # Create initial predictions (gi)
                for kf in active_filters:
                    kf.predict(fps)
                # Pass frame to YOLO model return face_frame
                if frame_num % 10 == 0 or frame_num == 1:
                    track_window = camera.predictFrame()

                    # If the YOLO model is detecting a face, call the Kalman filter class
                    if track_window.shape[0] > 0:
                        # For every box detected by YOLO, search to see if there is a Kalman filter nearby (gi)
                        for window in track_window:
                            x, y, w, h = window
                            best_match = None
                            nearest_match = 200

                            # For every filter, search for the closest one to the center of the detected box.
                            # If there is a match, update the filter. Otherwise, create a new filter. (gi)
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
                                new_filter = KalmanFilter(id, window)
                                active_filters.append(new_filter)
                                id += 1

                for kf in active_filters:
                    x, y, w, h = kf.get_position()
                    cv2.rectangle(frame, (int(x-(w//2)), int(y-(h//2))), (int(x + (w//2)), int(y + (h//2))), (0, 255, 0), 2)
                    ax, ay = kf.get_acceleration()
                    # After some rudementary testing, it seems that an acceleration of 30 or less is a good threshold for 
                    # determining if a face is still or not.
                    #print(kf.id, "Acceleration:", ax, ay)
                    if abs(ax) < 30 and abs(ay) < 30:
                        found_face = frame[int(y-(h//2)):int(y+(h//2)), int(x-(w//2)):int(x+(w//2))]
                        found_face = cv2.resize(found_face, (160, 160))
                        #TODO: Call the FaceNet model.


                # Display the frame
                cv2.imshow("Familiar Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

