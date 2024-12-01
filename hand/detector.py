import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

class HandDetector:
    def __init__(self, model_path):
        """Initialize hand detector"""
        self.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={
                'loss': custom_bbox_loss(width_weight=2.0),
                'iou_metric': iou_metric
            },
            safe_mode=False
        )
        self.input_size = (224, 224)
        self.cap = cv2.VideoCapture(0)
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, self.input_size)
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        # Add channel dimension
        processed = np.expand_dims(normalized, axis=-1)
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        return processed
        
    def draw_bbox(self, frame, bbox, color=(0, 255, 0)):
        """Draw bounding box on frame"""
        h, w = frame.shape[:2]
        x, y, box_w, box_h = bbox
        
        # Denormalize coordinates
        x = int(x * w)
        y = int(y * h)
        box_w = int(box_w * w)
        box_h = int(box_h * h)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 2)
        
        return frame
    
    def run(self):
        """Run the hand detector on webcam feed"""
        print("Starting hand detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Preprocess frame
            processed = self.preprocess_frame(frame)
            
            # Get prediction
            prediction = self.model.predict(processed, verbose=0)[0]
            
            # Draw bounding box
            frame = self.draw_bbox(frame, prediction)
            
            # Add text overlay
            cv2.putText(
                frame,
                "Hand Detection",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow('Hand Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

def custom_bbox_loss(width_weight=2.0):
    """Custom loss function with higher weight for width prediction"""
    def loss(y_true, y_pred):
        coord_loss = tf.reduce_mean(tf.square(y_true[:, :2] - y_pred[:, :2]))
        width_loss = width_weight * tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))
        height_loss = tf.reduce_mean(tf.square(y_true[:, 3] - y_pred[:, 3]))
        return coord_loss + width_loss + height_loss
    return loss

def iou_metric(y_true, y_pred):
    """IoU metric for model evaluation"""
    def calculate_iou(box1, box2):
        # Convert to [x1, y1, x2, y2] format
        b1_x1, b1_y1 = box1[0], box1[1]
        b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        b2_x1, b2_y1 = box2[0], box2[1]
        b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection
        x_left = max(b1_x1, b2_x1)
        y_top = max(b1_y1, b2_y1)
        x_right = min(b1_x2, b2_x2)
        y_bottom = min(b1_y2, b2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = b1_area + b2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    iou_scores = tf.py_function(
        lambda y_t, y_p: np.array([calculate_iou(t, p) for t, p in zip(y_t.numpy(), y_p.numpy())], dtype=np.float32),
        [y_true, y_pred],
        tf.float32
    )
    return tf.reduce_mean(iou_scores)

if __name__ == "__main__":
    model_path = "final_hand_detection_model.keras"  # Update this path if needed
    
    detector = HandDetector(model_path)
    detector.run()