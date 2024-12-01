import cv2
import os 
import time
import uuid 
import mediapipe as mp

def create_square_roi(frame, hand_landmarks, padding=40):
    """Create a square ROI around the hand"""
    height, width = frame.shape[:2]
    
    # Get hand bounding box
    x_coords = [landmark.x * width for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * height for landmark in hand_landmarks.landmark]
    x1, x2 = int(min(x_coords)), int(max(x_coords))
    y1, y2 = int(min(y_coords)), int(max(y_coords))
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    
    # Make square
    width = x2 - x1
    height = y2 - y1
    size = max(width, height)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Calculate new square coordinates
    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(frame.shape[1], x1 + size)
    y2 = min(frame.shape[0], y1 + size)
    
    return x1, y1, x2, y2

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

    # Create data directories
    IMAGES_PATH = os.path.join('data', 'images')
    os.makedirs(IMAGES_PATH, exist_ok=True)

    labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", 
              "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]
              
    print("\nControls:")
    print("- Press SPACE to capture an image")
    print("- Press N to move to next letter")
    print("- Press Q to quit")
    print("- Press D to delete last image")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_label_idx = 0
    while current_label_idx < len(labels):
        label = labels[current_label_idx]
        
        # Create directory for current label
        label_dir = os.path.join(IMAGES_PATH, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Count existing images for this label
        existing_images = len([f for f in os.listdir(label_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f'\nCollecting images for letter "{label}"')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            display_frame = frame.copy()
            hand_detected = False
            roi_coordinates = None
            
            # If hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(
                        display_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get and draw ROI
                    x1, y1, x2, y2 = create_square_roi(frame, hand_landmarks)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    hand_detected = True
                    roi_coordinates = (x1, y1, x2, y2)
            
            # Display info on frame
            cv2.putText(display_frame, f"Letter: {label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Images captured: {existing_images}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if not hand_detected:
                cv2.putText(display_frame, "No hand detected", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                print(f"Moving to next letter (captured {existing_images} images for {label})")
                current_label_idx += 1
                break
            elif key == ord(' ') and hand_detected and roi_coordinates:
                # Get hand ROI
                x1, y1, x2, y2 = roi_coordinates
                hand_roi = frame[y1:y2, x1:x2]
                
                if hand_roi.size > 0:
                    # Save image
                    imgname = os.path.join(label_dir, f"{label}.{str(uuid.uuid1())}.jpg")
                    cv2.imwrite(imgname, hand_roi)
                    existing_images += 1
                    print(f"Captured image {existing_images} for letter {label}")
                    time.sleep(0.2)
            elif key == ord('d') and existing_images > 0:
                # Get list of images and delete the last one
                images = sorted([f for f in os.listdir(label_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if images:
                    os.remove(os.path.join(label_dir, images[-1]))
                    existing_images -= 1
                    print(f"Deleted last image. {existing_images} images remaining for letter {label}")
    
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()