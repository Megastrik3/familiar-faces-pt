import math
import queue
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import kalman_filter as kalman_filter
import yolo_detect as yolo_det
import deepface_det as face_detect
import face_database as face_database
from camera_capture import CameraCapture


class FamiliarFacesPT():
    """This is the main class which is responsible for orcestrating the face detection, recognition, contact prompting
    and UI. The original code (as seen in previous github commits) was designed in native OpenCV. However, I was encountering
    issues when trying to prompt the user to add a new contact. The only way to display this in the UI was to overlay it on the live
    video feed. This resulted in the UI being captured by the face detection algorithm, and caused some strange behavior as a result.
    I knew that Tkinter was a python libaray used to create UIs in python, but I am not familiar with how to instantiate the UI or create new elements.
    So, I utilized AI to help me understand how I would need to rearchitect my main function to switch from OpenCV to Tkinter.
    (Please see linked chat: https://gemini.google.com/share/1a15f56a2803). I was able to reuse all of my original code (minus OpenCV UI stuff),
    and have made some changes to the way that the AI suggested I implement Tkinter. I fully understand how the tikinter code works.
    First we create the elements and window in the `__init__` function, then I create a function which runs the actual detection and display
    code. This is run inside the tkinter main loop, and updates every 15 miliseconds. This section is also responsible for displaying the contact creation
    window when there is a new contact ready to be added. Then, there are two functions which handle showing the naming prompt window, and saving the contact.
    The naming prompt window converts the contact img into the format compatible with tkinter, then unhides the contact name, and captures the user
    input. The save contact function uses the code I had already written to add the contact to the database, then hide the naming UI again.
    This is the single largest section of AI assisted code in my project, which is why I have left this massive (sorry :) ) comment explaining the code.
    """
    def __init__(self, root):
        self.root = root
        self.root.geometry("1920x1080")
        self.root.title("Familiar Faces (Prototype)")
        self.prompt_frame = tk.Frame(root)
        # Display live video feed
        self.live_feed_label = tk.Label(root, width=1280, height=720, bg="black")
        self.live_feed_label.pack(pady=10)

        # UI elements for name prompt and submission
        self.prompt_label = tk.Label(self.prompt_frame, text="New Face Detected! Enter Name:", font=("Arial", 14))
        self.prompt_label.pack(side=tk.LEFT, pady=5)
        self.name_entry = tk.Entry(self.prompt_frame, font=("Arial", 14))
        self.name_entry.pack(side=tk.LEFT, padx=5)

        self.submit_button = tk.Button(self.prompt_frame, text="Submit", command=self.save_contact) # confirm command
        self.submit_button.pack(side=tk.LEFT, padx=5)
        self.contact_image_label = tk.Label(self.prompt_frame)
        self.contact_image_label.pack(side=tk.LEFT, padx=10)

        # Define shared variables and start threads
        self.camera = CameraCapture()
        self.camera.start()
        self.yolo_model = yolo_det.YoloDetect()
        self.yolo_model.start(self.camera)
        self.id = 0
        self.frame_num = 0
        self.active_filters = []
        # Used VSCode CoPilot to suggest a fix for flickering boudning boxes.
        # It suggested that I save the results of the YOLO model and only run the kalman filter
        # updates if the YOLO model produces new results. This seems to have fixed the issue.
        # (It stated that the problem was a race condition between the YOLO thread and the main thread.
        # That is why it wasn't present in the previous version of the code, which ran the YOLO model in the main thread.)
        # (No link available for this conversation because it was using the built-in AI assistant)
        self.last_yolo_state = ""
        self.face_det = face_detect.DeepFaceDetector()
        self.pending_ui_contact_approvals = queue.Queue()
        self.face_det.start(self.pending_ui_contact_approvals)
        self.is_requesting_name = False
        self.user_provided_name = ""
        self.current_pending_contact = None
        self.face_db = face_database.Database()

        # Start video capture loop
        self.update_frame()

    def update_frame(self):
        frame = self.camera.get_frame()
        fps = self.camera.get_fps()
        has_valid_frame = frame is not None and frame.size > 0
        if has_valid_frame:
            self.frame_num += 1
            # Create initial predictions
            for kf in self.active_filters:
                kf.predict(fps)
            # Pass frame to YOLO model to find faces
            yolo_frames = self.yolo_model.get_latest_boxes()
            current_yolo_state = str(yolo_frames)
        

            # If the YOLO model found a face, create a new Kalman filter for it
            if current_yolo_state != self.last_yolo_state:
                # Get every detection box from YOLO output
                for window in yolo_frames:
                    x, y, w, h = window
                    best_match = None
                    nearest_match = 200

                    # For every filter, search for the closest filter to each detected face.
                    # If these is a match (within threshold), update the filter.
                    # Otherwise, create a new filter for the detected face.
                    for kf in self.active_filters:
                        cx, cy, _, _ = kf.get_position()
                        distance = math.hypot(x - cx, y - cy) # Calculate the L2 distance

                        if distance < nearest_match:
                            nearest_match = distance
                            best_match = kf

                    if best_match is not None:
                        best_match.update(window)
                    else:
                        new_filter = kalman_filter.KalmanFilter(self.id, window)
                        self.active_filters.append(new_filter)
                        self.id += 1
                self.last_yolo_state = current_yolo_state
            for kf in self.active_filters[:]:
                if kf.decay > 30:
                    self.active_filters.remove(kf)
            for kf in self.active_filters:
                x, y, w, h = kf.get_position()
                vx, vy = kf.get_velocity()
                self.camera.draw_box(frame, x, y, w, h, kf.getName())
                # After some rudementary testing, it seems that a velocity of 30 or less is a good threshold for 
                # determining if a face is still or not.
                #print(kf.id, "Acceleration:", ax, ay)
                if abs(vx) < 30 and abs(vy) < 30 and kf.permanance > 60:
                    self.face_det.detect(frame, x, y, w, h, kf)
                    self.camera.draw_box(frame, x, y, w, h, kf.getName())
                    kf.permanance = 0
            
            if not self.is_requesting_name and not self.pending_ui_contact_approvals.empty():
                self.current_pending_contact = self.pending_ui_contact_approvals.get()
                self.show_name_prompt_ui()
            
            cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2_frame = cv2.resize(cv2_frame, (1280, 720))
            img = Image.fromarray(cv2_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.live_feed_label.imgtk = imgtk
            self.live_feed_label.configure(image=imgtk)
        self.root.after(15, self.update_frame) # Schedule the next frame update

    def show_name_prompt_ui(self):
        self.is_requesting_name = True
        self.prompt_frame.pack(pady=10)

        if self.current_pending_contact["image"] is not None:
            face_img = self.current_pending_contact["image"]
            face_img = cv2.resize(face_img, (160, 160))
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)
            imgtk_face = ImageTk.PhotoImage(image=pil_face)
            self.contact_image_label.imgtk = imgtk_face
            self.contact_image_label.configure(image=imgtk_face)
        self.name_entry.focus_set()

    def save_contact(self):
        user_provided_name = self.name_entry.get().strip()
        if len(user_provided_name) > 0 and self.current_pending_contact:
            self.face_db.add_contact(user_provided_name, 
                    (self.current_pending_contact["embeddings"], self.current_pending_contact["created_at"]), 
                    self.current_pending_contact["image"])
            self.is_requesting_name = False # Done prompting!
            self.current_pending_contact = None

            self.name_entry.delete(0, tk.END)
            self.prompt_frame.pack_forget() # Hide the prompt UI

    def on_closing(self):
        self.camera.stop()
        self.yolo_model.stop()
        self.face_det.stop()
        self.root.destroy()

def main():
    # See large comment above.
    root = tk.Tk()
    app = FamiliarFacesPT(root)
    
    # Bind the 'Enter' key to the save_contact function
    root.bind('<Return>', lambda event: app.save_contact() if app.is_requesting_name else None)
    
    # Tell Tkinter what to do when the "X" button is clicked
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the Tkinter event loop
    root.mainloop()



if __name__ == "__main__":
    main()