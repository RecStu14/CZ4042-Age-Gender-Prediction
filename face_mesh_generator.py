import face_recognition as fr
import cv2
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import glob
import os
import dlib

# IMPORTING LIBRARIES
import mediapipe as mp

def obtain_face_mesh(FILEPATH, FILENAME):

    # INITIALIZING OBJECTS
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # DETECT THE FACE LANDMARKS
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            image = cv2.imread(f"{FILEPATH}/{FILENAME}")

            # Flip the image horizontally and convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Detect the face landmarks
            results = face_mesh.process(image)

            # To improve performance
            image.flags.writeable = True

            # Convert back to the BGR color space
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            template_img = cv2.imread("blackimage_200x200.jpg")
            
            # Draw the face mesh annotations on the image.
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
            # # Save the mesh
            cv2.imwrite(f"data/utk_dataset/processed/{FILENAME}.jpg", template_img)
      


