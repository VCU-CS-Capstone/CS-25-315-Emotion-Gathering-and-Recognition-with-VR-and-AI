import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import json
import os
from pathlib import Path

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

model_path = 'face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

image = cv2.imread('PrivateTest_88305.jpg')
if image is None:
    raise Exception("Could not load image")

# Convert to RGB (MediaPipe expects RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, 
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5,  # Try adjusting these thresholds
    num_faces = 1,
    min_tracking_confidence	= 0.5,
    min_face_presence_confidence = 0.5)

def save_blendshapes(face_landmarker_result, filename):
    """
    Save only the blendshapes data to a JSON file
    """
    if face_landmarker_result.face_blendshapes:
        blendshapes_data = []
        for blendshape in face_landmarker_result.face_blendshapes[0]:  # Get first face's blendshapes
            blendshapes_data.append({
                'category': blendshape.category_name,
                'score': float(blendshape.score)  # Convert to float for JSON serialization
            })
        
        with open(filename, 'w') as f:
            json.dump(blendshapes_data, f, indent=2)
        print(f"Blendshapes saved to {filename}")
    else:
        print("No blendshapes detected")

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
    
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        face_landmarker_result = landmarker.detect(mp_image)
        
    if not face_landmarker_result.face_landmarks:
         print("No faces detected in the image")
    else:
        print(f"Detected {len(face_landmarker_result.face_landmarks)} faces")
        annotated_image = draw_landmarks_on_image(image_rgb, face_landmarker_result)
            
            # Display the image
        cv2.imshow('Face Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
        print(face_landmarker_result)

    if not face_landmarker_result.face_blendshapes:
            print("No blendshapes detected")
    else:
            # Save the blendshapes
            save_blendshapes(face_landmarker_result, 'blendshapes.json')

except Exception as e:
    print(f"Error during face detection: {str(e)}")

