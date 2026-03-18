from deepface import DeepFace
import numpy as np
import threading

class DeepFaceDetector():
    def __init__(self):
        self.model_name = "Facenet"
        self.face_database = {}
        self.thread = None


    def detect(self, img):
        self.thread = threading.Thread(target=self._detect_face, args=(img,))
        self.thread.start()


    def _detect_face(self, img):
        self.embedding = DeepFace.represent(img, model_name=self.model_name, detector_backend='mtcnn')[0]["embedding"]
        self.embedding = self.embedding / np.linalg.norm(self.embedding)
        print("Embedding shape:", self.embedding.shape)