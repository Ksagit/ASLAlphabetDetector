import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import cv2

def load_dataset(data_dir, target_size=(128, 128), validation_split=0.2, test_split=0.1):
    """Load dataset from the augmented_images directory with balanced class splits"""
    images = []
    labels = []
    label_to_index = {}
    current_label = 0
    
    print("Loading images...")
    
    for letter in 'abcdefghiklmnopqrstuvwxy':
        folder_path = os.path.join(data_dir, letter)
        if not os.path.isdir(folder_path):
            print(f"Warning: Directory not found for letter {letter}")
            continue
            
        label_to_index[letter] = current_label
        print(f"Processing class: {letter}")
        
        image_files = os.listdir(folder_path)
        if len(image_files) != 133:
            print(f"Warning: Expected 133 images for letter {letter}, found {len(image_files)}")
        
        for image_file in image_files:
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize image
            resized = cv2.resize(gray, target_size)
            
            # Normalize pixel values to [0,1]
            normalized = resized.astype('float32') / 255.0
            
            images.append(normalized)
            labels.append(current_label)
            
        current_label += 1
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Reshape images to include channel dimension
    X = X.reshape(X.shape[0], target_size[0], target_size[1], 1)
    
    # Split per class to maintain proportions
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    for class_idx in range(len(label_to_index)):
        # Get indices for this class
        class_indices = np.where(y == class_idx)[0]
        class_X = X[class_indices]
        class_y = y[class_indices]
        
        # Calculate split sizes for this class
        n_samples = len(class_indices)
        n_test = int(test_split * n_samples)
        n_val = int(validation_split * n_samples)
        
        # Split into train, validation, and test
        X_class_train = class_X[:-n_test-n_val]
        y_class_train = class_y[:-n_test-n_val]
        
        X_class_val = class_X[-n_test-n_val:-n_test]
        y_class_val = class_y[-n_test-n_val:-n_test]
        
        X_class_test = class_X[-n_test:]
        y_class_test = class_y[-n_test:]
        
        # Append to main lists
        X_train.append(X_class_train)
        y_train.append(y_class_train)
        X_val.append(X_class_val)
        y_val.append(y_class_val)
        X_test.append(X_class_test)
        y_test.append(y_class_test)
    
    # Concatenate all splits
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_to_index

def create_model(input_shape, num_classes):
    """Create a CNN model for gesture classification"""
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Fourth Convolutional Block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),  
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, label_mapping, epochs=20):
    """Train the model and display results"""
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = len(label_mapping)
    
    # Create model
    model = create_model(input_shape, num_classes)
    
    # Compile model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
        
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # Evaluate model
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create reverse mapping (index to label)
    index_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [index_to_label[i] for i in sorted(index_to_label.keys())]
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Save final model
    try:
        model.save('gesture_classifier_model.keras')
        print("\nModel saved successfully as 'gesture_classifier_model.keras'")
    except Exception as e:
        print(f"Error saving full model: {e}")
    
    return model, history

if __name__ == "__main__":
    data_dir = os.path.join('data', 'augmented_images')
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_mapping = load_dataset(data_dir)
        
    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val, X_test, y_test, label_mapping)