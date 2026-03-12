import cv2
import numpy as np
import threading

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
    

if __name__ == "__main__":
    camera = CameraCapture()
    camera.start()

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow("Familiar Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

