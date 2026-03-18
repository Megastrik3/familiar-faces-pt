from deepface import DeepFace
import numpy as np
import threading
import cv2

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

    def detect(self, img, x, y, w, h, adjust=0):
        self.frame = img
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        face= self.crop_face(img, x, y, w, h, adjust)
        self.thread = threading.Thread(target=self._detect_face, args=(face,))
        self.thread.start()


    def _detect_face(self, img):
        try:
            self.embedding = DeepFace.represent(img, model_name=self.model_name, detector_backend='mtcnn')[0]["embedding"]
            self.embedding = self.embedding / np.linalg.norm(self.embedding)
            print("Embedding shape:", self.embedding.shape)
            self._searchDatabase(self.embedding, img)
            self.retry = 0
        except Exception as e:
            print("Face not properly centered or detected")
            self.retry += 1
            if self.retry < 3:
                self.detect(self.frame, self.x, self.y, self.w, self.h, adjust=20)



    def _searchDatabase(self, embedding, img):
        min_distance = float('inf')
        identity = None
        if not self.embedded_face_database:
            # save the face image for later
            self.face_img_database[f"unknown{self.id}"] = img, self.id
            self.embedded_face_database[f"unknown{self.id}"] = embedding, self.id, 1
            print(f"Hello unknown{self.id}! You've been seen 1 time.")
            self.id += 1
        else:
            for name, data in self.embedded_face_database.items():
                db_embedding, id, quantity = data
                distance = 1 - np.dot(embedding, db_embedding)
                print(f"Comparing with {name}, distance: {distance}")
                if distance < min_distance:
                    min_distance = distance
                    identity = name
            if min_distance < 0.75:
                print(f"Hello {identity}! You've been seen {self.embedded_face_database[identity][2]+1} times.")
                self.embedded_face_database[identity] = (embedding, self.embedded_face_database[identity][1], self.embedded_face_database[identity][2]+1)
            else:
                # save the face image for later
                self.face_img_database[f"unknown{self.id}"] = img, self.id
                self.embedded_face_database[f"unknown{self.id}"] = embedding, self.id, 1
                print(f"Hello unknown{self.id}! You've been seen 1 time.")
                self.id += 1
