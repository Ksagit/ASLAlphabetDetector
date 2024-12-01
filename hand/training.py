import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import cv2
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def apply_augmentation(image, bbox):
    """Apply augmentations that preserve bounding box coordinates"""
    augmented_images = []
    augmented_boxes = []
    
    # Original image and box
    augmented_images.append(image)
    augmented_boxes.append(bbox)
    
    # Horizontal flip
    flipped_image = cv2.flip(image, 1)
    flipped_bbox = [1 - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]  # Adjust x coordinate
    augmented_images.append(flipped_image)
    augmented_boxes.append(flipped_bbox)
    
    # Brightness variations
    for factor in [0.8, 1.2]:  # Darker and brighter
        adjusted = np.clip(image * factor, 0, 1)
        augmented_images.append(adjusted)
        augmented_boxes.append(bbox)  # Bbox unchanged
    
    # Contrast adjustment
    for factor in [0.8, 1.2]:  # Lower and higher contrast
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = np.clip((image - mean) * factor + mean, 0, 1)
        augmented_images.append(adjusted)
        augmented_boxes.append(bbox)  # Bbox unchanged
        
    # Small rotations (only small angles to prevent bbox distortion)
    for angle in [-10, 10]:  # -10 and +10 degrees
        center = (image.shape[1] / 2, image.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Adjust bbox for rotation
        # Convert bbox to corners
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        corners = corners * np.array([image.shape[1], image.shape[0]])
        
        # Rotate corners
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        transformed = M.dot(corners_ones.T).T
        
        # Get new bbox
        min_xy = transformed.min(axis=0)
        max_xy = transformed.max(axis=0)
        new_bbox = [
            min_xy[0] / image.shape[1],
            min_xy[1] / image.shape[0],
            (max_xy[0] - min_xy[0]) / image.shape[1],
            (max_xy[1] - min_xy[1]) / image.shape[0]
        ]
        
        # Only add if bbox is still fully within image
        if (new_bbox[0] >= 0 and new_bbox[1] >= 0 and
            new_bbox[0] + new_bbox[2] <= 1 and
            new_bbox[1] + new_bbox[3] <= 1):
            augmented_images.append(rotated)
            augmented_boxes.append(new_bbox)
    
    return augmented_images, augmented_boxes

def load_hand_dataset(dataset_path, image_size=(224, 224)):
    """Load hand detection dataset with augmentations"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations.json"
    
    images = []
    boxes = []
    
    print("Loading dataset...")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    total_images = len(annotations['images'])
    print(f"Found {total_images} images in annotations")
    
    # Process each image and its annotation
    for idx, img_ann in enumerate(annotations['images'], 1):
        try:
            # Load image
            image_path = images_dir / img_ann['file_name']
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Preprocess image
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize
            
            # Get bbox
            bbox = img_ann['bbox']
            
            # Apply augmentations
            aug_images, aug_boxes = apply_augmentation(image, bbox)
            
            images.extend(aug_images)
            boxes.extend(aug_boxes)
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{total_images} images")
                
        except Exception as e:
            print(f"Error processing {img_ann['file_name']}: {str(e)}")
            continue
    
    if len(images) == 0:
        raise ValueError("No images loaded from dataset")
    
    print(f"Successfully loaded {len(images)} images (including augmentations)")
    return np.array(images), np.array(boxes)

def create_model(input_shape=(224, 224, 3)):
    """Create a CNN model for hand detection"""
    # Load pre-trained MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Base model
        base_model,
        
        # Additional layers for hand detection
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='sigmoid')  # bbox coordinates [x, y, width, height]
    ])
    
    return model

def train_model(dataset_path, epochs=50, batch_size=32):
    """Train the hand detection model"""
    # Load dataset
    images, boxes = load_hand_dataset(dataset_path)
    
    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, boxes, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['accuracy', 'mae']
    )
    
    # Create callbacks with adjusted parameters
    callbacks = [
        # Save best model weights
        tf.keras.callbacks.ModelCheckpoint(
            'best_hand_detection_model.keras',
            save_best_only=True,
            save_weights_only=True,  # Only save weights
            monitor='val_loss',
            verbose=1
        ),
        # Reduce learning rate with adjusted parameters
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, history

def predict_on_image(model, image_path, image_size=(224, 224)):
    """Make prediction on a single image"""
    # Read and preprocess image
    image = cv2.imread(image_path)
    image_display = image.copy()
    image = cv2.resize(image, image_size)
    image = image / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    
    # Denormalize coordinates
    height, width = image_display.shape[:2]
    x, y, w, h = prediction
    x = int(x * width)
    y = int(y * height)
    w = int(w * width)
    h = int(h * height)
    
    # Draw bounding box
    cv2.rectangle(image_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image_display

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Set path to your dataset
    dataset_path = "hand_dataset"  # Update this path if needed
    
    # Train model
    model, training_history = train_model(dataset_path)
    
    # Save final model with complete architecture and weights
    model.save('final_hand_detection_model.keras', save_format='keras_v3')
