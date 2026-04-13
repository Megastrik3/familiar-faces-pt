***
# Familiar Faces - Prototype

> *A discrete, AI-powered prototype designed to help you remember the names of the people you meet.*

---

## Problem Statement
Remembering people's names has always been a bit of a challenge for me. Either I don't hear their name properly the first time, or I only see someone once or twice and don't see them frequently enough to remember their name. 

This is an especially difficult challenge for teachers, recruiters, and those who meet lots of people on a regular basis. It can be challenging to recall names quickly when you've already met 10+ people at that same time or when someone has a unique name. 

**Familiar Faces aims to solve this problem.** What if there was a tool whose sole purpose was to remember names for you? That is exactly what Familiar Faces is: a tool designed to discretely help you remember people's names so that you can better relate and communicate with your peers, colleagues, and students.

## Project Description
Familiar Faces is a prototype for a mobile and AR/XR application designed to help users remember the names of people they have not seen in a while or have only recently met. 

It works by detecting the faces of people you encounter. After a defined number of encounters—or after a specific time duration—Familiar Faces will ask if you want to add that person's face to your **"Contacts"**. Once a person is added, their name will be displayed above their head the next time you see them (after a set duration of time).

## Proposed Method
To achieve this, several AI models are utilized. While the application is designed with mobile hardware constraints in mind, the current computer-based prototype will forgo these constraints as necessary for rapid development.

Processing every camera frame is computationally expensive. Therefore, we use a pipeline of "filters" to ensure heavy detection only runs when necessary:

1. **Pre-Detection & Cropping:** A lightweight **YOLO** model quickly and cheaply detects every face that appears in the frame. This avoids running complex recognition on empty frames and crops the faces out for the next step.
2. **Object Tracking:** A **Kalman filter** tracks the movement of the detected face throughout the frame, capturing frames where the face is clearly visible. It also anchors the 2D spatial projection (the text box displaying the name) above the contact's head. *(Note: Future work will explore 3D spatial projection).*
3. **Facial Recognition:** The cropped face image is passed to Google's **FaceNet** algorithm. Optimized for mobile devices, FaceNet encodes the detected face into a high-dimensional vector and saves it to a database. 
4. **Identification:** A simple distance calculation checks if this face embedding is "familiar" (already in the database) or new.

## Data Sources
This application primarily relies on pre-trained models for detecting faces and embedding them into high-dimensional vectors. Because faces are learned over time via the app, there is no need to pre-train FaceNet on a defined set of user faces. 

However, to optimize face detection:
* The YOLO model will be trained locally using the [**WIDER FACE Kaggle dataset**](https://www.kaggle.com/datasets/canomercik/wider-face-dataset-for-yolov12-format). 

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