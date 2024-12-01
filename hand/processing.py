import cv2
import json
import numpy as np
from pathlib import Path
import os

def apply_augmentation(image, bbox):
    """Apply augmentations that preserve bounding box coordinates"""
    augmented_data = []  # List to store (image, bbox, suffix) tuples
    
    # Original image and box
    augmented_data.append((image, bbox, "original"))
    
    # Horizontal flip
    flipped_image = cv2.flip(image, 1)
    flipped_bbox = [1 - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]
    augmented_data.append((flipped_image, flipped_bbox, "flipped"))
    
    # Brightness variations
    for factor, suffix in zip([0.8, 1.2], ["dark", "bright"]):
        adjusted = np.clip(image * factor, 0, 1)
        augmented_data.append((adjusted, bbox, suffix))
    
    # Contrast adjustment
    for factor, suffix in zip([0.8, 1.2], ["lowcontrast", "highcontrast"]):
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = np.clip((image - mean) * factor + mean, 0, 1)
        augmented_data.append((adjusted, bbox, suffix))
    
    # Small rotations
    for angle, suffix in zip([-10, 10], ["rotm10", "rotp10"]):
        center = (image.shape[1] / 2, image.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Adjust bbox for rotation
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        corners = corners * np.array([image.shape[1], image.shape[0]])
        
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        transformed = M.dot(corners_ones.T).T
        
        min_xy = transformed.min(axis=0)
        max_xy = transformed.max(axis=0)
        new_bbox = [
            min_xy[0] / image.shape[1],
            min_xy[1] / image.shape[0],
            (max_xy[0] - min_xy[0]) / image.shape[1],
            (max_xy[1] - min_xy[1]) / image.shape[0]
        ]
        
        if (new_bbox[0] >= 0 and new_bbox[1] >= 0 and
            new_bbox[0] + new_bbox[2] <= 1 and
            new_bbox[1] + new_bbox[3] <= 1):
            augmented_data.append((rotated, new_bbox, suffix))
    
    return augmented_data

def create_augmented_dataset(input_path, output_path):
    """Create augmented dataset and save to disk"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directories
    output_images_dir = output_path / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original annotations
    with open(input_path / "annotations.json", 'r') as f:
        annotations = json.load(f)
    
    # Prepare new annotations
    new_annotations = {"images": []}
    
    print("Processing images...")
    total_images = len(annotations['images'])
    
    for idx, img_ann in enumerate(annotations['images'], 1):
        try:
            # Load image
            image_path = input_path / "images" / img_ann['file_name']
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Get original image dimensions
            original_height, original_width = image.shape[:2]
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            
            # Get bbox
            bbox = img_ann['bbox']
            
            # Apply augmentations
            augmented_data = apply_augmentation(image, bbox)
            
            # Save augmented images and create annotations
            base_name = Path(img_ann['file_name']).stem
            
            for aug_image, aug_bbox, suffix in augmented_data:
                # Create new filename
                new_filename = f"{base_name}_{suffix}.jpg"
                
                # Save image (convert back to uint8)
                save_path = output_images_dir / new_filename
                cv2.imwrite(str(save_path), (aug_image * 255).astype(np.uint8))
                
                # Add annotation
                new_annotations['images'].append({
                    "file_name": new_filename,
                    "bbox": aug_bbox,
                    "original_file": img_ann['file_name'],
                    "augmentation": suffix
                })
            
            print(f"Processed {idx}/{total_images} images")
            
        except Exception as e:
            print(f"Error processing {img_ann['file_name']}: {str(e)}")
            continue
    
    # Save new annotations
    with open(output_path / "annotations.json", 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"\nAugmented dataset creation completed:")
    print(f"- Input images: {total_images}")
    print(f"- Output images: {len(new_annotations['images'])}")
    print(f"- Output directory: {output_path}")

if __name__ == "__main__":
    input_dataset = "hand_dataset"
    output_dataset = "hand_dataset_augmented"
    
    create_augmented_dataset(input_dataset, output_dataset)