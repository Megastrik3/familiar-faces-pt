import math

import cv2
import kalman_filter as kalman_filter
import yolo_detect as yolo_det
import deepface_det as face_det
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
    
    def draw_box(self, x, y, w, h):
        """Draw a bounding box on the camera frame.

        Args:
            frame (array): The camera frame to draw on.
            x (int): The x coordinate of the top left corner of the bounding box.
            y (int): The y coordinate of the top left corner of the bounding box.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.
        """
        cv2.rectangle(self.frame, (int(x-(w//2)), int(y-(h//2))), (int(x + (w//2)), int(y + (h//2))), (0, 255, 0), 2)

    



if __name__ == "__main__":
    camera = CameraCapture()
    camera.start()
    id = 0
    frame_num = 0
    active_filters = []
    face_det = face_det.DeepFaceDetector()

    try:
        while True:
            fps = camera.get_fps() # Get FPS from cv2
            frame = camera.get_frame() # Get frame from camera
            has_valid_frame = frame is not None and frame.size > 0

            if not has_valid_frame:
                # Camera thread may not have produced a frame yet.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            if has_valid_frame:
                frame_num += 1
                # Create initial predictions
                for kf in active_filters:
                    kf.predict(fps)
                # Pass frame to YOLO model to find faces
                if frame_num % 10 == 0 or frame_num == 1:
                    track_window = camera.predictFrame()

                    # If the YOLO model found a face, create a new Kalman filter for it
                    if len(track_window) > 0:
                        # Get every detection box from YOLO output
                        for window in track_window:
                            x, y, w, h = window
                            best_match = None
                            nearest_match = 200

                            # For every filter, search for the closest filter to each detected face.
                            # If these is a match (within threshold), update the filter.
                            # Otherwise, create a new filter for the detected face.
                            for kf in active_filters:
                                cx, cy, _, _ = kf.get_position()
                                distance = math.hypot(x - cx, y - cy) # Calculate the L2 distance

                                if distance < nearest_match:
                                    nearest_match = distance
                                    best_match = kf
                                # If the Kalman filter has not been updated for `5 frames, prune it
                                if kf.decay > 15:
                                    active_filters.remove(kf)
                            if best_match is not None:
                                best_match.update(window)
                            else:
                                new_filter = kalman_filter.KalmanFilter(id, window)
                                active_filters.append(new_filter)
                                id += 1
            for kf in active_filters:
                x, y, w, h = kf.get_position()
                ax, ay = kf.get_acceleration()
                camera.draw_box(x, y, w, h)
                # After some rudementary testing, it seems that an acceleration of 30 or less is a good threshold for 
                # determining if a face is still or not.
                #print(kf.id, "Acceleration:", ax, ay)
                if abs(ax) < 30 and abs(ay) < 30 and kf.permanance > 30:
                    #found_face = camera.crop_face(x, y, w, h)
                    #TODO: Call the FaceNet model. 
                    face_det.detect(frame, x, y, w, h)
                    kf.permanance = 0

            # Display the frame once a non-empty image is available.
            cv2.imshow("Familiar Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Exiting...")
        camera.stop()
    except Exception as e:
        print("An error occurred:", e)
    finally:
        # Exit the camera capture
        camera.stop()
