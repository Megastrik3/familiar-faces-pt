# Class was generated with AI. All functions were read through and understood by the author.
# The code follows the design standard desird for this project. Multithreadding is a challenging topic, 
# which is why AI was used to to assist in the development of this class.
import cv2
import threading


class CameraCapture:
    """Create a camera frame stream on a dedicated thread.
    """
    def __init__(self, camera_index=1):
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
            cv2.destroyAllWindows()

    def get_frame(self):
        """Get the most recent frame from the video stream.

        Returns:
            Tuple: Camera frame.
        """
        return self.frame
    
    def get_fps(self):
        """Get camera FPS

        Returns:
            int: Camera FPS
        """
        return self.capture.get(cv2.CAP_PROP_FPS)
    
    def crop_face(self, x, y, w, h):
        """Crop a face from the camera frame.

        Args:
            frame (array): The camera frame to crop from.
            x (int): The x coordinate of the top left corner of the bounding box.
            y (int): The y coordinate of the top left corner of the bounding box.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.
        Returns:
            array: The cropped face.
        """
        return cv2.resize(self.frame[(y-(h//2)):y+(h//2), (x-(w//2)):x+(w//2)], (160, 160))
    

    
    def draw_box(self, frame, x, y, w, h, active_name=None):
        """Draw a bounding box on the camera frame.

        Args:
            frame (array): The camera frame to draw on.
            x (int): The x coordinate of the top left corner of the bounding box.
            y (int): The y coordinate of the top left corner of the bounding box.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.
        """
        #cv2.rectangle(frame, (int(x-(w//2)), int(y-(h//2))), (int(x + (w//2)), int(y + (h//2))), (0, 255, 0), 2)
        if active_name is not None and active_name != "None":
            cv2.putText(frame, f"{active_name}", (int(x-(w//2)), int(y-(h//2))-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)