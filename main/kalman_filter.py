import cv2
import numpy as np

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
