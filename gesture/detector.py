import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp

def preprocess_frame(frame, target_size=(128, 128)):
    """Preprocess a frame the same way as training data"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization with reduced clip limit
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Very light blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.3)
        
        # Resize image
        resized = cv2.resize(blurred, target_size)
        
        # Normalize pixel values to [0,1]
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        processed = normalized.reshape(1, target_size[0], target_size[1], 1)
        
        return processed
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Load the trained model
    try:
        model = tf.keras.models.load_model('gesture_classifier_model.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: Could not load model: {e}")
        return

    # Load label mapping
    try:
        data = np.load('preprocessed_data.npz', allow_pickle=True)
        label_mapping = data['label_mapping'].item()
        # Create reverse mapping (index to label)
        index_to_label = {v: k for k, v in label_mapping.items()}
    except:
        print("Error: Could not load label mapping from preprocessed_data.npz")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set smaller resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create window
    cv2.namedWindow('Hand Gesture Classifier', cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save a frame")
    print("- Press 'r' to reset confidence display")

    # Variables for smoothing predictions
    prediction_history = []
    history_length = 8  # Increased for more stable predictions
    confidence_threshold = 0.6  # Reduced since our model is more optimized
    min_prediction_count = 4  # Minimum number of same predictions needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_detected = False
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Calculate hand bounding box
                x_coords = [landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # Add padding to bounding box
                padding = 40
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                # Ensure square ROI
                width = x2 - x1
                height = y2 - y1
                size = max(width, height)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                x1 = max(0, center_x - size // 2)
                y1 = max(0, center_y - size // 2)
                x2 = min(frame.shape[1], x1 + size)
                y2 = min(frame.shape[0], y1 + size)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                hand_detected = True
                
                # Extract and preprocess ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:  # Check if ROI is not empty
                    try:
                        processed_roi = preprocess_frame(roi)
                        
                        # Make prediction
                        predictions = model.predict(processed_roi, verbose=0)
                        
                        # Get top 3 predictions
                        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                        top_3_confidences = predictions[0][top_3_indices]
                        top_3_letters = [index_to_label[idx] for idx in top_3_indices]

                        # Add current prediction to history (keep using the top prediction for stability)
                        current_pred_idx = top_3_indices[0]
                        current_confidence = top_3_confidences[0]
                        
                        prediction_history.append((current_pred_idx, current_confidence))
                        if len(prediction_history) > history_length:
                            prediction_history.pop(0)
                        
                        # Get most common prediction from history
                        if prediction_history:
                            pred_indices, confidences = zip(*prediction_history)
                            unique_preds, counts = np.unique(pred_indices, return_counts=True)
                            most_common_idx = unique_preds[np.argmax(counts)]
                            most_common_count = counts[np.argmax(counts)]
                            
                            # Only show prediction if we have enough consistent predictions
                            if most_common_count >= min_prediction_count:
                                # Calculate average confidence for the most common prediction
                                avg_confidence = np.mean([conf for idx, conf in prediction_history 
                                                        if idx == most_common_idx])
                                
                                if avg_confidence > confidence_threshold:
                                    predicted_letter = index_to_label[most_common_idx]
                                    
                                    # Main prediction display
                                    display_text = f"Predicted: {predicted_letter} ({avg_confidence:.2f})"
                                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                                    
                                    # Draw background rectangle for main prediction
                                    cv2.rectangle(frame, 
                                                (x1, y1-10-text_size[1]-10),
                                                (x1+text_size[0]+10, y1-10),
                                                (0, 255, 0),
                                                -1)
                                    
                                    # Draw main prediction text
                                    cv2.putText(frame, display_text, 
                                              (x1+5, y1-15), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                              (0, 0, 0), 2)
                                    
                                    # Draw stability indicator
                                    stability = most_common_count / len(prediction_history)
                                    stability_text = f"Stability: {stability:.2f}"
                                    cv2.putText(frame, stability_text,
                                              (x1+5, y1-40),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                              (0, 255, 0), 2)
                                    
                                    # Draw top 3 predictions in the corner
                                    padding = 10
                                    bg_color = (50, 50, 50)
                                    text_color = (255, 255, 255)
                                    
                                    # Draw background for top 3
                                    cv2.rectangle(frame,
                                                (padding, padding),
                                                (200, 110),
                                                bg_color,
                                                -1)
                                    
                                    # Draw "Top 3 Predictions" header
                                    cv2.putText(frame,
                                              "Top 3 Predictions:",
                                              (padding + 5, padding + 20),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.6,
                                              text_color,
                                              1)
                                    
                                    # Draw top 3 predictions
                                    for i, (letter, conf) in enumerate(zip(top_3_letters, top_3_confidences)):
                                        text = f"{i+1}. {letter}: {conf:.2f}"
                                        cv2.putText(frame,
                                                  text,
                                                  (padding + 5, padding + 45 + i*20),
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.5,
                                                  text_color,
                                                  1)
                                    
                    except Exception as e:
                        print(f"Error processing ROI: {e}")
        
        if not hand_detected:
            prediction_history = []  # Reset predictions when no hand is detected
                
        # Show frame
        cv2.imshow('Hand Gesture Classifier', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'captured_frame_{timestamp}.jpg', frame)
            print(f"Saved frame as captured_frame_{timestamp}.jpg")
        elif key == ord('r'):
            # Reset prediction history
            prediction_history = []
            print("Reset prediction history")

    # Clean up
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()