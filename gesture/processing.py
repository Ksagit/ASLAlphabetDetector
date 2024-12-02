import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def apply_augmentation(image, target_size, save_dir, original_filename, label):
    """Apply various augmentation techniques to a single image and save them"""
    augmented_images = []
    
    # Create directory for augmented images if it doesn't exist
    aug_dir = os.path.join(save_dir, label)
    os.makedirs(aug_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(original_filename)[0]
    
    # Original image
    augmented_images.append(image)
    # Save original
    orig_path = os.path.join(aug_dir, f"{base_filename}_orig.jpg")
    cv2.imwrite(orig_path, (image * 255).astype(np.uint8))
    
    # Rotation variations
    for i, angle in enumerate([-8, -4, 4, 8]):
        rotated = rotate(image, angle, reshape=False)
        rotated = np.clip(rotated, 0, 1)
        rotated = cv2.resize(rotated, target_size)
        augmented_images.append(rotated)
        # Save rotated image
        rot_path = os.path.join(aug_dir, f"{base_filename}_rot{angle}.jpg")
        cv2.imwrite(rot_path, (rotated * 255).astype(np.uint8))
    
    # Random shifts
    for i in range(2):
        tx = np.random.uniform(-1.5, 1.5)
        ty = np.random.uniform(-1.5, 1.5)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(image, translation_matrix, target_size)
        augmented_images.append(shifted)
        # Save shifted image
        shift_path = os.path.join(aug_dir, f"{base_filename}_shift{i}.jpg")
        cv2.imwrite(shift_path, (shifted * 255).astype(np.uint8))
    
    # Random zoom
    for i in range(2):
        zoom_factor = np.random.uniform(0.97, 1.03)
        zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
        zoomed = cv2.resize(zoomed, target_size)
        augmented_images.append(zoomed)
        # Save zoomed image
        zoom_path = os.path.join(aug_dir, f"{base_filename}_zoom{i}.jpg")
        cv2.imwrite(zoom_path, (zoomed * 255).astype(np.uint8))
    
    # Brightness variations
    for i, factor in enumerate([0.92, 1.08]):
        brightened = np.clip(image * factor, 0, 1)
        augmented_images.append(brightened)
        # Save brightness-adjusted image
        bright_path = os.path.join(aug_dir, f"{base_filename}_bright{i}.jpg")
        cv2.imwrite(bright_path, (brightened * 255).astype(np.uint8))
    
    return np.array(augmented_images)

def load_and_preprocess_images(data_dir, target_size=(128, 128), validation_split=0.2, test_split=0.1):
    """
    Load images, preprocess them, apply augmentation, and split into train/validation/test sets
    """
    images = []
    labels = []
    label_to_index = {}
    current_label = 0
    
    # Create augmented data directory
    augmented_data_dir = os.path.join('data', 'augmented_images')
    os.makedirs(augmented_data_dir, exist_ok=True)

    print("Loading and preprocessing images...")
    
    # Load and preprocess images
    for label_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
            
        label_to_index[label_folder] = current_label
        print(f"Processing class: {label_folder}")
        
        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(folder_path, image_file)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply mild adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Very light blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.3)
            
            # Resize image
            resized = cv2.resize(blurred, target_size)
            
            # Normalize pixel values to [0,1]
            normalized = resized.astype('float32') / 255.0
            
            # Apply augmentation and save images
            augmented_batch = apply_augmentation(normalized, target_size, 
                                               augmented_data_dir, image_file, 
                                               label_folder)
            
            # Add all augmented versions to dataset
            images.extend(augmented_batch)
            labels.extend([current_label] * len(augmented_batch))
            
        current_label += 1
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Reshape images to include channel dimension
    X = X.reshape(X.shape[0], target_size[0], target_size[1], 1)
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=validation_split, random_state=42)
    
    # Create index to label mapping
    index_to_label = {v: k for k, v in label_to_index.items()}
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total images after augmentation: {len(X)}")
    print(f"Training images: {len(X_train)}")
    print(f"Validation images: {len(X_val)}")
    print(f"Test images: {len(X_test)}")
    print(f"Number of classes: {len(label_to_index)}")
    print("\nClass distribution:")
    for label, index in label_to_index.items():
        count = np.sum(y == index)
        print(f"{label}: {count} images")
        
    print(f"\nAugmented images have been saved to: {augmented_data_dir}")
    
    # Visualize some preprocessed images
    plt.figure(figsize=(15, 8))
    plt.suptitle("Original and Augmented Images Examples")
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i].reshape(target_size), cmap='gray')
        plt.title(f"{index_to_label[y_train[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_to_index

if __name__ == "__main__":
    data_dir = os.path.join('data', 'images')
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_mapping = load_and_preprocess_images(data_dir)