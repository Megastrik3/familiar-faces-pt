***
# Familiar Faces - Prototype

> *A discrete, AI-powered prototype designed to help you remember the names of the people you meet.*

---

## Problem Statement
Remembering people's names has always been a bit of a challenge for me. Either I don't hear their name properly the first time, or I only see someone once or twice and don't see them frequently enough to remember their name. This is especially a difficult challenge for teachers, recruiters, and those who meet lots of people on a regular basis. It can be challenging to recall names quickly when you've already met 10+ people at that same time or when someone has a unique name. Familiar Faces aims to solve this problem. What if there was a tool that's sole purpose was to remember names for you? That is exactly what Familiar Faces is, a tool designed to discretely help you remember people's names so that you can better relate and communicate with your peers, colleagues, and students.

## Project Description
Familiar faces (prototype) is a prototype for a mobile and AR/XR application designed to help users remember names of people that they have not seen in a while or that they have only recently met. It does this by detecting faces of people that you encounter. Then, after a defined number of encounters, or after a time duration, Familiar Faces will ask the user if they want to add that person's face to their list of "Contacts". Once a person is added to the user's "contacts," they will have their name displayed above their head next time the user sees them after a set duration of time.

## Proposed Method
In order to achieve this goal, several AI models will be used. Since this is an application designed to run on mobile hardware but is being prototyped on a computer, the application will be designed with the hardware constraints of a mobile device in mind, but will forgo these design constraints as necessary to complete the project on time.

Processing every frame captured by a camera is a very computationally expensive task. As such, various "filters" will need to be used to run detection only when a face is actively present. To do this, a lightweight YOLO model will be used to simply detect every face that appears in the frame. Using this method, we will be able to identify faces quickly and cheaply, without having to run the face detection algorithm on every frame. This pre-detection step will also be used to crop the faces out so that the face is ready to be passed to the facial recognition model. Then, a kalman filter will be used to track the movement of the detected face throughout the frame. This will be used to capture frames where the face is clearly visible. The cropped face image will then be passed to the face detection algorithm. The kalman filter will also be how the name is displayed above the contact's head. Once a person has been positively identified, a textbox will be displayed in 2D space above their head (future work will look into 3d spatial projection, but that is outside of the scope of the project for now). In order to identify a person by their face, the FaceNet algorithm by Google will be used. This algorithm is already optimized for running on mobile devices, making it a great choice for this application. The FaceNet model works by encoding detected faces into a high-dimensional vector, then saving that vector to a database. A simple distance calculation can then be used to see if this face embedding is "familiar" or not.

## Data Sources
This model uses primarily pretrained models for detecting people's faces and embedding them into high-dimensional vectors. Additionally, because faces are learned over time, there is no need to train the FaceNet model on a defined set of faces. However, in order to learn more about model training and to gain more experience in training YOLO models, the YOLO model used to detect faces will be trained locally using the [WIDER FACE](https://www.kaggle.com/datasets/canomercik/wider-face-dataset-for-yolov12-format) dataset. 

## Project Structure
The repository is split into two primary components: the core application module (`main`) and the YOLO model training/evaluation space (`YOLO`).

```text
├── main/                       # Core application scripts
│   ├── __main__.py             # Entry point for the application
│   ├── camera_capture.py       # Handles camera feed and video processing
│   ├── deepface_det.py         # Handles facial recognition and embeddings (FaceNet)
│   ├── face_database.py        # Manages the database of face vectors
│   ├── kalman_filter.py        # Object tracking for continuous face positioning
│   └── yolo_detect.py          # YOLO face detection integration
├── YOLO/                       # YOLO model training directory
│   ├── yolo-training.ipynb     # Jupyter notebook for training the face detection model
│   ├── dvc.yaml                # Data Version Control configuration
│   ├── yolo26n.pt              # Base YOLO weights
│   └── runs/                   # Training results, validation metrics, and fine-tuned weights
├── devenv.nix / devenv.yaml    # Declarative development environment configurations
└── .dvc/                       # Data Version Control setup
```

## How to Run

### Training the YOLO Model
If you wish to train the YOLO model locally from scratch:
1. Download the **WIDER FACE** dataset formatted for YOLOv12 from Kaggle.
2. Create a folder named `datasets` inside the `YOLO/` directory.
3. Extract the downloaded dataset into the newly created `YOLO/datasets/` folder.
4. Open `YOLO/yolo-training.ipynb` in your preferred Jupyter environment.
5. Run each cell sequentially to begin training.

### Running the Application
To run the main tracking and recognition prototype:
1. Ensure your camera is connected and available.
2. From the root directory of the project, execute the main module:

```bash
python -m main.__main__
```
*(Alternatively, you can run the `__main__.py` file directly from within the `main` directory).*

_Note: This README was stylized using AI based on the original README and additional program execution instructions_