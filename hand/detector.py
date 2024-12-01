import tensorflow as tf
import cv2
import numpy as np

def preprocess_frame(frame, target_size=(224, 224)):
    """Preprocess frame for model input"""
    # Resize frame
    frame_resized = cv2.resize(frame, target_size)
    # Normalize
    frame_normalized = frame_resized / 255.0
    return frame_normalized

def draw_bbox(frame, bbox):
    """Draw bounding box on frame"""
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    
    # Denormalize coordinates
    x = int(x * width)
    y = int(y * height)
    w = int(w * width)
    h = int(h * height)
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

def main():
    # Load model
    model = tf.keras.models.load_model('final_hand_detection_model.keras')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Create copy for display
        display_frame = frame.copy()
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Get prediction
        prediction = model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)[0]
        
        # Draw bounding box
        display_frame = draw_bbox(display_frame, prediction)
        
        # Add text showing confidence
        cv2.putText(
            display_frame,
            f"Hand Detection",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow('Hand Detection', display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()