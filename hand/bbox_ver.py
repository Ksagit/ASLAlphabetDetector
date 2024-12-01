import cv2
import json
from pathlib import Path
import numpy as np

def verify_annotations(dataset_path):
    """Display images with their bounding boxes from annotations"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations.json"
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations['images'])} images in annotations")
    current_idx = 0
    
    while current_idx < len(annotations['images']):
        # Get current image data
        img_ann = annotations['images'][current_idx]
        image_path = images_dir / img_ann['file_name']
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            current_idx += 1
            continue
            
        # Get bbox coordinates
        bbox = img_ann['bbox']
        height, width = image.shape[:2]
        
        # Convert normalized coordinates to pixel values
        x = int(bbox[0] * width)
        y = int(bbox[1] * height)
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)
        
        # Draw bounding box
        image_with_box = image.copy()
        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text with image info
        info_text = f"Image: {img_ann['file_name']} ({current_idx + 1}/{len(annotations['images'])})"
        bbox_text = f"bbox: x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]:.2f}, h={bbox[3]:.2f}"
        
        cv2.putText(image_with_box, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image_with_box, bbox_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow('Annotation Verification', image_with_box)
        
        # Handle key presses
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('n') or key == ord(' '):  # Next image
            current_idx = min(current_idx + 1, len(annotations['images']) - 1)
        elif key == ord('p') or key == ord('b'):  # Previous image
            current_idx = max(current_idx - 1, 0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = "hand_dataset_augmented"
    verify_annotations(dataset_path)
    
print("""
Controls:
- Space or 'n': Next image
- 'p' or 'b': Previous image
- 'q': Quit
""")