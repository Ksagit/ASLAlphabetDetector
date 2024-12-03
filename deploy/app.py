import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template
import threading
from queue import Queue

def custom_bbox_loss(width_weight=2.0):
    def loss(y_true, y_pred):
        coord_loss = tf.reduce_mean(tf.square(y_true[:, :2] - y_pred[:, :2]))
        width_loss = width_weight * tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))
        height_loss = tf.reduce_mean(tf.square(y_true[:, 3] - y_pred[:, 3]))
        return coord_loss + width_loss + height_loss
    return loss

def calculate_iou(y_true, y_pred):
    iou_scores = tf.py_function(
        lambda y_t, y_p: np.array([_calculate_single_iou(t, p) for t, p in zip(y_t.numpy(), y_p.numpy())], dtype=np.float32),
        [y_true, y_pred],
        tf.float32
    )
    return tf.reduce_mean(iou_scores)

def _calculate_single_iou(y_true, y_pred):
    true_x1, true_y1 = y_true[0], y_true[1]
    true_x2, true_y2 = y_true[0] + y_true[2], y_true[1] + y_true[3]
    
    pred_x1, pred_y1 = y_pred[0], y_pred[1]
    pred_x2, pred_y2 = y_pred[0] + y_pred[2], y_pred[1] + y_pred[3]
    
    x_left = max(true_x1, pred_x1)
    y_top = max(true_y1, pred_y1)
    x_right = min(true_x2, pred_x2)
    y_bottom = min(true_y2, pred_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    union_area = true_area + pred_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

app = Flask(__name__)

# Load models with custom objects
custom_objects = {
    'loss': custom_bbox_loss(2.0),
    'iou_metric': calculate_iou
}

hand_model = tf.keras.models.load_model('final_hand_detection_model.keras', 
                                       custom_objects=custom_objects,
                                       safe_mode=False)
gesture_model = tf.keras.models.load_model('gesture_classifier_model.keras',
                                          safe_mode=False)
# Global variables for thread-safe frame sharing
frame_queue = Queue(maxsize=1)
processed_frames = {
    'hand_detection': Queue(maxsize=1),
    'gesture_prediction': Queue(maxsize=1)
}

def process_frame_for_hand(frame, confidence_threshold=0.5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    
    bbox = hand_model.predict(input_tensor, verbose=0)[0]
    
    # Add confidence score calculation (example - adjust based on your model)
    confidence = np.mean(bbox)  # Or use a more sophisticated confidence calculation
    
    if confidence < confidence_threshold:
        return frame.copy(), None
    
    h, w = frame.shape[:2]
    x = int(bbox[0] * w)
    y = int(bbox[1] * h)
    width = int(bbox[2] * w)
    height = int(bbox[3] * h)
    
    output_frame = frame.copy()
    cv2.rectangle(output_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(output_frame, f"Conf: {confidence:.2f}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return output_frame, (x, y, width, height) if confidence >= confidence_threshold else None

def process_frame_for_gesture(frame, bbox, confidence_threshold=0.7):
    if bbox is None:
        return frame
        
    x, y, width, height = bbox
    hand_region = frame[y:y+height, x:x+width]
    if hand_region.size == 0:
        return frame
    
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    
    prediction = gesture_model.predict(input_tensor, verbose=0)
    confidence = np.max(prediction)
    
    if confidence < confidence_threshold:
        return frame
        
    gesture_class = np.argmax(prediction)
    letters = 'abcdefghiklmnopqrstuvwxy'
    predicted_letter = letters[gesture_class]
    
    output_frame = frame.copy()
    cv2.putText(output_frame, f"{predicted_letter} ({confidence:.2f})", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return output_frame

def process_frame_for_gesture(frame, bbox):
    x, y, width, height = bbox
    
    # Extract hand region
    hand_region = frame[y:y+height, x:x+width]
    if hand_region.size == 0:
        return frame
    
    # Preprocess for gesture classification
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    
    # Get prediction
    prediction = gesture_model.predict(input_tensor, verbose=0)
    gesture_class = np.argmax(prediction)
    
    # Map class index to letter (adjust based on your model's classes)
    letters = 'abcdefghiklmnopqrstuvwxy'
    predicted_letter = letters[gesture_class]
    
    # Draw prediction
    output_frame = frame.copy()
    cv2.putText(output_frame, f"Gesture: {predicted_letter}", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return output_frame

def capture_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update raw frame
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
        
        # Process for hand detection
        hand_frame, bbox = process_frame_for_hand(frame)
        if processed_frames['hand_detection'].full():
            processed_frames['hand_detection'].get()
        processed_frames['hand_detection'].put(hand_frame)
        
        # Process for gesture recognition only if hand detected
        gesture_frame = frame.copy()
        if bbox is not None:
            gesture_frame = process_frame_for_gesture(hand_frame, bbox)
        if processed_frames['gesture_prediction'].full():
            processed_frames['gesture_prediction'].get()
        processed_frames['gesture_prediction'].put(gesture_frame)

def generate_frames(queue_name=None):
    while True:
        if queue_name is None:
            frame = frame_queue.get()
        else:
            frame = processed_frames[queue_name].get()
            
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    if feed_type == 'raw':
        return Response(generate_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif feed_type == 'hand':
        return Response(generate_frames('hand_detection'),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    elif feed_type == 'gesture':
        return Response(generate_frames('gesture_prediction'),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start frame capture thread
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(debug=False)