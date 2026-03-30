from deepface import DeepFace
from deepface.modules.exceptions import FaceNotDetected
from collections import Counter
import face_database as face_db
import numpy as np
import threading
import cv2
import queue

class DeepFaceDetector():
    def __init__(self):
        self.model_name = "Facenet"
        self.embedded_face_database = {}
        self.face_img_database = {}
        self.thread = None
        self.id = len(self.embedded_face_database)
        self.frame = None
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.retry = 0
        self.return_name = None
        self.db = face_db.Database()

        # The following variables and start, stop, and _process_queue functions were added 
        # with the help of Google Gemini. I was having some threading issues with the original code
        # and being new to multithreading, used AI to help me understand how to properly use threading to run the face detection in the background.
        # This is part of a larger re-threading conversation that I had with Gemini. I understand how the code works and find it very interesting
        # how important queing and execution order are for this application.
        # https://gemini.google.com/share/f693e2b7fa8c
        self.queue = queue.Queue()
        self.running = False
        self.recan_count = 0
        self.rescan_threshold = 13
        self.pending_ui_queue = None

    def start(self, pending_ui_queue):
        self.pending_ui_queue = pending_ui_queue
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_queue)
            self.thread.start()


    def crop_face(self, img, x, y, w, h, adjust=0):
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
        return cv2.resize(img[(y-(h+adjust)//2):y+(h+adjust)//2, (x-(w+adjust)//2):x+(w+adjust)//2], (160, 160))

    def detect(self, img, x, y, w, h, kf, adjust=0):
        self.frame = img
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        face = self.crop_face(img, x, y, w, h, adjust)
        self.queue.put((face, kf))

    def _process_queue(self):
        while self.running:
            try:
                face, kf = self.queue.get()
                self._detect_face(face, kf)

                if self.recan_count >= self.rescan_threshold:
                    pending_contacts = self.db.rescan_unknowns()
                    print(len(pending_contacts))
                    for contact in pending_contacts:
                        print(f"Re-scanned contacts")
                        self.pending_ui_queue.put(contact)
                        self.db.clean_unknowns(contact['embeddings'])
                    self.recan_count = 0

                self.queue.task_done()
                self.recan_count += 1
            except queue.Empty:
                continue


    def _detect_face(self, img, kf):
        try:
            # Generate embedding for the detected face
            self.embedding = DeepFace.represent(img, model_name=self.model_name, detector_backend='mtcnn')[0]["embedding"]
            # Normalize the embedding
            self.embedding = self.embedding / np.linalg.norm(self.embedding)
            # Run KNN to identify face
            self.return_name = self.knn_locate_names(self.embedding, img)

            if self.return_name is not None:
                kf.setName(self.return_name)
            # Reset retry counter on successful detection
            self.retry = 0
        except FaceNotDetected as e:
            print("Face not properly centered or detected")
            print("Error:", e)
            self.retry += 1
            if self.retry < 3:
                self.detect(self.frame, self.x, self.y, self.w, self.h, kf, adjust=20)

    def knn_locate_names(self, embedding, img):
        # Get all contacts from the database
        contacts = self.db.get_contacts()
        # Get all embeddings from the database. Has form (contact_ids(array), embedding(array))
        contact_ids, embeddings = self.db.get_embeddings()
        # If there are embeddings in the database
        if len(embeddings) != 0:
            # Compute cosine similarity between the new embedding and all embeddings in the database
            similarities = np.dot(embeddings, embedding)
            distances = 1 - similarities
            # Get the indices of the top 5 most similar embeddings
            sorted_similarities = np.argsort(distances)[:5]
            # Get the distances for the top 5 most similar embeddings
            top_distances = [distances[idx] for idx in sorted_similarities]
            avg_distance = np.mean(top_distances)

            # Define a similarity threshold. If the most common contact_id is below this threshold, consider it a match.
            if avg_distance < 0.35:
                # Get the most common contact_id among the top 5 most similar embeddings
                most_common = Counter([contact_ids[idx] for idx in sorted_similarities]).most_common(1)[0][0] # Get the first most common, then get the value of that most common
                print(most_common)
                # Walk through the contacts to find the contact info corresponding to the contact_id
                # This is AI code. I want to completely rewrite this function
                contact_info = next((contact for contact in contacts if contact[1] == most_common), None)
                if contact_info:
                    name, _, encounter_count, _, _ = contact_info
                    print(f"Hello {name}! You've been seen {encounter_count+1} times.")
                    self.db.update_contact(int(most_common), embedding)
                    return name
        print("No matching contact found. Consider adding this face to the database.")
        self.db.add_unknown(embedding, img)
        return None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()