import cv2
import mediapipe as mp
import json
import os
from pathlib import Path
import numpy as np
import time

class WebcamDatasetCreator:
    def __init__(self, output_dir="hand_dataset", capture_interval=0.5):
        """Initialize the dataset creator
        
        Args:
            output_dir (str): Directory to save the dataset
            capture_interval (float): Minimum seconds between captures
        """
        # Create dataset directories
        self.dataset_dir = Path(output_dir)
        self.images_dir = self.dataset_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Initialize variables
        self.image_counter = 0
        self.annotations = {"images": []}
        self.last_capture_time = 0
        self.capture_interval = capture_interval
    
    def get_hand_bbox(self, hand_landmarks, image_shape):
        """Convert MediaPipe hand landmarks to normalized bounding box"""
        x_coords = []
        y_coords = []
        
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Get bounding box coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 0.1
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - width * padding)
        x_max = min(1, x_max + width * padding)
        y_min = max(0, y_min - height * padding)
        y_max = min(1, y_max + height * padding)
        
        # Return normalized [x, y, width, height]
        return [
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min
        ]
    
    def save_image_and_annotation(self, image, bbox):
        """Save image and its annotation"""
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create image filename
        image_name = f"hand_{str(self.image_counter).zfill(4)}.jpg"
        image_path = self.images_dir / image_name
        
        # Save grayscale image
        cv2.imwrite(str(image_path), gray_image)
        
        # Create annotation
        annotation = {
            "file_name": image_name,
            "bbox": bbox
        }
        self.annotations["images"].append(annotation)
        
        # Save updated annotations
        with open(self.dataset_dir / "annotations.json", 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        self.image_counter += 1
        return gray_image
    
    def run(self):
        """Run the dataset creator"""
        print("Starting dataset creation...")
        print("Press 'q' to quit")
        print("Hand detection is running automatically")
        
        try:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Failed to capture frame")
                    break
                
                # Flip image horizontally for a later selfie-view display
                image = cv2.flip(image, 1)
                
                # Convert the BGR image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect hands
                results = self.hands.process(image_rgb)
                
                # Create display image (grayscale but converted to BGR for drawing colored landmarks)
                display_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                
                # Draw hand landmarks and check for valid detection
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            display_image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Check if enough time has passed since last capture
                        current_time = time.time()
                        if current_time - self.last_capture_time >= self.capture_interval:
                            # Get bounding box
                            bbox = self.get_hand_bbox(hand_landmarks, image.shape)
                            
                            # Save image and annotation
                            self.save_image_and_annotation(image, bbox)
                            self.last_capture_time = current_time
                            
                            # Draw bbox on display
                            h, w = display_image.shape[:2]
                            x, y = int(bbox[0] * w), int(bbox[1] * h)
                            width, height = int(bbox[2] * w), int(bbox[3] * h)
                            cv2.rectangle(display_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Add text overlay
                cv2.putText(
                    display_image,
                    f"Images captured: {self.image_counter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Show image
                cv2.imshow('Hand Dataset Creator', display_image)
                
                # Check for quit
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print(f"\nDataset creation completed:")
            print(f"- Total images captured: {self.image_counter}")
            print(f"- Dataset saved in: {self.dataset_dir}")

if __name__ == "__main__":
    # Create and run dataset creator
    creator = WebcamDatasetCreator(
        output_dir="hand_dataset",
        capture_interval=0.5  # Capture every 0.5 seconds when hand is detected
    )
    creator.run() 