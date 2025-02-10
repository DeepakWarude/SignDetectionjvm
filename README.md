# SignDetectionjvm
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import time

# Configuration
CLASS_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def create_dataset_structure():
    """Create the necessary directory structure for the dataset"""
    base_dir = os.path.join(os.getcwd(), 'dataset')
    train_dir = os.path.join(base_dir, 'train')
    
    # Create main directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    
    # Create subdirectories for each sign
    for label in CLASS_LABELS:
        label_dir = os.path.join(train_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Created directory: {label_dir}")
    
    return base_dir

def collect_training_data():
    """Collect training data using webcam"""
    base_dir = os.path.join(os.getcwd(), 'dataset', 'train')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create a window
    cv2.namedWindow('Data Collection', cv2.WINDOW_NORMAL)
    
    for label in CLASS_LABELS:
        label_dir = os.path.join(base_dir, label)
        count = 0
        
        print(f"\nCollecting data for sign: {label}")
        print("Instructions:")
        print("- Press 'c' to capture an image")
        print("- Press 'q' to skip to next sign")
        print("- Press 'ESC' to exit")
        
        capture_delay = 0  # To prevent multiple captures
        
        while count < 30:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                continue
            
            # Draw a rectangle in the center as a guide
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            
            # Add text overlays
            cv2.putText(frame, f"Sign: {label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images: {count}/30", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):  # Skip to next sign
                print(f"Skipping sign {label}")
                break
            elif key == 27:  # ESC to exit
                print("Exiting data collection")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('c') and time.time() - capture_delay > 1.0:  # Capture with 1-second delay
                # Create filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{label}_{timestamp}_{count}.jpg"
                filepath = os.path.join(label_dir, filename)
                
                # Save the image
                try:
                    cv2.imwrite(filepath, frame)
                    print(f"Captured image {count+1}/30 for sign {label}")
                    count += 1
                    capture_delay = time.time()
                    
                    # Flash effect
                    flash = np.ones(frame.shape, dtype=np.uint8) * 255
                    cv2.imshow('Data Collection', flash)
                    cv2.waitKey(50)
                except Exception as e:
                    print(f"Error saving image: {e}")
            
            # Show success message when done with current sign
            if count >= 30:
                print(f"Completed collecting images for sign {label}")
                time.sleep(1)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection completed!")

def preprocess_data(data_dir):
    """Preprocess the data using ImageDataGenerator"""
    print("Starting data preprocessing...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    try:
        train_data = datagen.flow_from_directory(
            data_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        val_data = datagen.flow_from_directory(
            data_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        print("Data preprocessing completed successfully!")
        return train_data, val_data
        
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        return None, None

def main():
    print("\n=== Sign Language Detection System ===")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Start real-time detection")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                print("\nInitializing data collection...")
                base_dir = create_dataset_structure()
                collect_training_data()
            elif choice == '2':
                # Training code (same as before)
                pass
            elif choice == '3':
                # Detection code (same as before)
                pass
            elif choice == '4':
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")
        
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
