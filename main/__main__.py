import math
import queue

import cv2
import kalman_filter as kalman_filter
import yolo_detect as yolo_det
import deepface_det as face_detect
import face_database as face_database
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
        cv2.rectangle(frame, (int(x-(w//2)), int(y-(h//2))), (int(x + (w//2)), int(y + (h//2))), (0, 255, 0), 2)
        if active_name is not None and active_name != "None":
            cv2.putText(frame, f"Still Face Detected: {active_name}", (int(x-(w//2)), int(y-(h//2))-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    camera = CameraCapture()
    camera.start()
    yolo_model = yolo_det.YoloDetect()
    yolo_model.start(camera)
    id = 0
    frame_num = 0
    active_filters = []
    # Used VSCode CoPilot to suggest a fix for flickering boudning boxes.
    # It suggested that I save the results of the YOLO model and only run the kalman filter
    # updates if the YOLO model produces new results. This seems to have fixed the issue.
    # (It stated that the problem was a race condition between the YOLO thread and the main thread.
    # That is why it wasn't present in the previous version of the code, which ran the YOLO model in the main thread.)
    # (No link available for this conversation because it was using the built-in AI assistant)
    last_yolo_state = ""
    face_det = face_detect.DeepFaceDetector()
    pending_ui_contact_approvals = queue.Queue()
    face_det.start(pending_ui_contact_approvals)
    is_requsting_name = False
    user_provided_name = ""
    current_pending_contact = None
    face_db = face_database.Database()

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
                yolo_frames = yolo_model.get_latest_boxes()
                current_yolo_state = str(yolo_frames)
            

                # If the YOLO model found a face, create a new Kalman filter for it
                if current_yolo_state != last_yolo_state:
                    # Get every detection box from YOLO output
                    for window in yolo_frames:
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

                        if best_match is not None:
                            best_match.update(window)
                        else:
                            new_filter = kalman_filter.KalmanFilter(id, window)
                            active_filters.append(new_filter)
                            id += 1
                    last_yolo_state = current_yolo_state
                for kf in active_filters[:]:
                    if kf.decay > 30:
                        active_filters.remove(kf)
            for kf in active_filters:
                x, y, w, h = kf.get_position()
                vx, vy = kf.get_velocity()
                camera.draw_box(frame, x, y, w, h, kf.getName())
                # After some rudementary testing, it seems that a velocity of 30 or less is a good threshold for 
                # determining if a face is still or not.
                #print(kf.id, "Acceleration:", ax, ay)
                if abs(vx) < 30 and abs(vy) < 30 and kf.permanance > 60:
                    #found_face = camera.crop_face(x, y, w, h)
                    #TODO: Call the FaceNet model. 
                    name = face_det.detect(frame, x, y, w, h, kf)
                    camera.draw_box(frame, x, y, w, h, kf.getName())
                    kf.permanance = 0

############### AI CODE ##################################

            if not is_requsting_name and not pending_ui_contact_approvals.empty():
                current_pending_contact = pending_ui_contact_approvals.get()
                is_requsting_name = True
                user_provided_name = ""
            if is_requsting_name:
                # Draw a dark background box for readability
                cv2.rectangle(frame, (10, 10), (450, 100), (0, 0, 0), -1)
                cv2.putText(frame, "New Face Detected! Enter Name:", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display what the user is currently typing
                cv2.putText(frame, f"> {user_provided_name}_", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Optional: Draw the cropped face of the unknown person in the top right corner
                # so the user knows who they are naming!
                if current_pending_contact["image"] is not None:
                    face_img = current_pending_contact["image"]
                    frame[10:170, -170:-10] = cv2.resize(face_img, (160, 160))


            # Display the frame once a non-empty image is available.
            cv2.imshow("Familiar Faces", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and not is_requsting_name: # Only allow quitting if not in the middle of naming a new contact
                break
    # 3. Handle Keystrokes if prompting
            if is_requsting_name and key != 255: # 255 means no key was pressed
                if key == 13 or key == 10: # Enter key (13 on Windows, 10 on Linux/Mac)
                    if len(user_provided_name) > 0:
                        # Save the contact!
                        face_db.add_contact(user_provided_name, 
                                            current_pending_contact["embeddings"], 
                                            current_pending_contact["image"])
                        
                        # You would also run your DELETE FROM unknown_contacts SQL here
                        
                        is_requsting_name = False # Done prompting!
                        current_pending_contact = None
                
                elif key == 8 or key == 127: # Backspace key
                    user_provided_name = user_provided_name[:-1]
                    
                elif 32 <= key <= 126: # Valid ASCII characters (letters, numbers, space)
                    user_provided_name += chr(key)

        #################### END AI CODE ##################################
    except KeyboardInterrupt:
        print("Exiting...")
        camera.stop()
    except Exception as e:
        print("An error occurred:", e)
    finally:
        # Exit the camera capture
        camera.stop()
        yolo_model.stop()
        face_det.stop()



if __name__ == "__main__":
    main()