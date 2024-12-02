import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calculate_iou(y_true, y_pred):
    """Calculate Intersection over Union between boxes"""
    # Convert from [x, y, w, h] to [x1, y1, x2, y2]
    true_x1 = y_true[0]
    true_y1 = y_true[1]
    true_x2 = y_true[0] + y_true[2]
    true_y2 = y_true[1] + y_true[3]
    
    pred_x1 = y_pred[0]
    pred_y1 = y_pred[1]
    pred_x2 = y_pred[0] + y_pred[2]
    pred_y2 = y_pred[1] + y_pred[3]
    
    # Calculate intersection
    x_left = max(true_x1, pred_x1)
    y_top = max(true_y1, pred_y1)
    x_right = min(true_x2, pred_x2)
    y_bottom = min(true_y2, pred_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    union_area = true_area + pred_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def custom_bbox_loss(width_weight=2.0):
    """Custom loss function with higher weight for width prediction"""
    def loss(y_true, y_pred):
        # Coordinate loss (x, y)
        coord_loss = tf.reduce_mean(tf.square(y_true[:, :2] - y_pred[:, :2]))
        
        # Width loss (weighted)
        width_loss = width_weight * tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))
        
        # Height loss
        height_loss = tf.reduce_mean(tf.square(y_true[:, 3] - y_pred[:, 3]))
        
        # Total loss
        total_loss = coord_loss + width_loss + height_loss
        return total_loss
    return loss

def iou_metric(y_true, y_pred):
    """IoU metric for model evaluation"""
    iou_scores = tf.py_function(
        lambda y_t, y_p: np.array([calculate_iou(t, p) for t, p in zip(y_t.numpy(), y_p.numpy())], dtype=np.float32),
        [y_true, y_pred],
        tf.float32
    )
    return tf.reduce_mean(iou_scores)

def load_hand_dataset(dataset_path, image_size=(224, 224)):
    """Load hand detection dataset without augmentations"""
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
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Preprocess image
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
            
            # Get bbox
            bbox = img_ann['bbox']
            
            images.append(image)
            boxes.append(bbox)
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{total_images} images")
                
        except Exception as e:
            print(f"Error processing {img_ann['file_name']}: {str(e)}")
            continue
    
    if len(images) == 0:
        raise ValueError("No images loaded from dataset")
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(boxes)

def create_model():
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Add more specific detection layers
    model = models.Sequential([
        layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, 
                    activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, 
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(4, 
                    activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])
    return model

def train_model(dataset_path, epochs=100, batch_size=32):
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=custom_bbox_loss(width_weight=2.0),
        metrics=['mae', iou_metric]
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_hand_detection_model.keras',
            save_best_only=True,
            save_weights_only=True,
            monitor='val_iou_metric',
            mode='max',
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
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
    plt.figure(figsize=(12, 4))

    # Plot IoU
    plt.subplot(1, 2, 1)
    plt.plot(history.history['iou_metric'], label='Training IoU')
    plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.title('Model IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
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

def plot_predictions(model, dataset_path, num_examples=4, image_size=(224, 224)):
    """Plot model predictions vs ground truth"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations.json"
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4*num_examples))
    fig.suptitle('Model Predictions vs Ground Truth', fontsize=16)
    
    # Randomly select images
    selected_images = np.random.choice(annotations['images'], num_examples, replace=False)
    
    for idx, img_ann in enumerate(selected_images):
        image_path = images_dir / img_ann['file_name']
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
            
        # Ground truth
        true_bbox = img_ann['bbox']
        h, w = image.shape[:2]
        true_x = int(true_bbox[0] * w)
        true_y = int(true_bbox[1] * h)
        true_w = int(true_bbox[2] * w)
        true_h = int(true_bbox[3] * h)
        
        ground_truth = image.copy()
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(ground_truth, (true_x, true_y), 
                     (true_x + true_w, true_y + true_h), 
                     (0, 255, 0), 2)
        
        # Model prediction
        image_resized = cv2.resize(image, image_size)
        image_normalized = (image_resized / 255.0).astype(np.float32)
        image_normalized = np.expand_dims(image_normalized, axis=-1)  # Add channel dimension
        prediction = model.predict(np.expand_dims(image_normalized, axis=0), verbose=0)[0]
        
        pred_x = int(prediction[0] * w)
        pred_y = int(prediction[1] * h)
        pred_w = int(prediction[2] * w)
        pred_h = int(prediction[3] * h)
        
        prediction_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(prediction_image, (pred_x, pred_y), 
                     (pred_x + pred_w, pred_y + pred_h), 
                     (0, 0, 255), 2)
        
        # Calculate IoU for this prediction
        iou = calculate_iou(true_bbox, prediction)
        
        # Plot
        axes[idx, 0].imshow(ground_truth[..., ::-1])  # Convert BGR to RGB for matplotlib
        axes[idx, 0].set_title(f'Ground Truth')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(prediction_image[..., ::-1])  # Convert BGR to RGB for matplotlib
        axes[idx, 1].set_title(f'Prediction (IoU: {iou:.2f})')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.close()

def predict_on_image(model, image_path, image_size=(224, 224)):
    """Make prediction on a single image"""
    # Read and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_display = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    # Preprocess for prediction
    image_resized = cv2.resize(image, image_size)
    image_normalized = (image_resized / 255.0).astype(np.float32)
    image_normalized = np.expand_dims(image_normalized, axis=-1)  # Add channel dimension
    
    # Make prediction
    prediction = model.predict(np.expand_dims(image_normalized, axis=0), verbose=0)[0]
    
    # Denormalize coordinates
    height, width = image.shape[:2]
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
    dataset_path = "hand_dataset_augmented"
    
    # Train model
    model, training_history = train_model(dataset_path)
    
    # Save final model
    model.save('final_hand_detection_model.keras', save_format='keras_v3')
    
    # Plot prediction examples
    plot_predictions(model, dataset_path)