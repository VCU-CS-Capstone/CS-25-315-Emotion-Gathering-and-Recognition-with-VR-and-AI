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


def process_images_recursively(input_folder, output_folder):
    """
    Process images in all subfolders recursively
    """
    # Create base output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Setup MediaPipe
    model_path = 'face_landmarker.task'
    
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.3,
        num_faces=1,
        min_tracking_confidence=0.3,
        min_face_presence_confidence=0.3
    )

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    total_processed = 0
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(input_folder):
            # Create corresponding output subdirectory
            relative_path = os.path.relpath(root, input_folder)
            current_output_folder = os.path.join(output_folder, relative_path)
            Path(current_output_folder).mkdir(parents=True, exist_ok=True)
            
            # Process images in current directory
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    print(f"\nProcessing {os.path.join(relative_path, file)}...")
                    
                    # Construct full file paths
                    input_path = os.path.join(root, file)
                    output_blendshapes_path = os.path.join(
                        current_output_folder, 
                        os.path.splitext(file)[0] + '_blendshapes.json'
                    )
                    
                    try:
                        # Load and convert image
                        image = cv2.imread(input_path)
                        if image is None:
                            print(f"Could not load {file}")
                            continue
                        
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        # Detect faces
                        face_landmarker_result = landmarker.detect(mp_image)
                        
                        if not face_landmarker_result.face_landmarks:
                            print(f"No faces detected in {file}")
                            continue
                        
                        # Save blendshapes
                        if face_landmarker_result.face_blendshapes:
                            save_blendshapes(face_landmarker_result, output_blendshapes_path)
                            total_processed += 1
                        else:
                            print(f"No blendshapes detected in {file}")
                        
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        continue

        cv2.destroyAllWindows()
        print("\nProcessing complete!")
        print(f"Processed {total_processed} images")

# Use the function
if __name__ == "__main__":
    input_folder = "test"  # your main folder with emotion subfolders
    output_folder = "output_blendshapes"  # base output folder
    
    process_images_recursively(input_folder, output_folder)
