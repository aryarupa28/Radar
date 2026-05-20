import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical

def create_sequences(features, labels, seq_len=10):
    """
    Creates sequences of length seq_len from the extracted features.
    """
    X_seq = []
    y_seq = []
    
    # Simple sliding window
    for i in range(len(features) - seq_len + 1):
        X_seq.append(features[i : i + seq_len])
        # The label for the sequence is the label of the last frame (or majority)
        # Here we assume if the sequence contains an accident frame, it's an accident.
        # But looking at mapping.csv, 0s are No Accident, 1s are Accident.
        # We will take the label of the last frame to detect "onset" or status at t.
        y_seq.append(labels[i + seq_len - 1])
        
    return np.array(X_seq), np.array(y_seq)

def extract_features_vgg16(image_folder, mapping_csv, base_model):
    """
    Extracts features from images using VGG16.
    """
    data = pd.read_csv(mapping_csv)
    images = []
    labels = []
    
    
    # Check if images exist, if not, try to extract from video
    first_image_path = os.path.join(image_folder, "0.jpg")
    if not os.path.exists(first_image_path):
        print("Images not found. Attempting to extract from video...")
        # Assume video is in the same folder, named 'Accidents.mp4' based on file list
        video_path = os.path.join(image_folder, "Accidents.mp4")
        if os.path.exists(video_path):
            extract_frames_from_video(video_path, image_folder)
        else:
             print(f"Warning: Video file {video_path} not found. Cannot extract frames.")

    print("Loading images...")

    for index, row in data.iterrows():
        img_id = row['Image_ID']
        label = row['Class']
        img_path = os.path.join(image_folder, img_id)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)
        else:
            print(f"Warning: {img_path} not found.")

    images = np.array(images)
    labels = np.array(labels)
    
    print("Preprocessing input...")
    images = preprocess_input(images)
    
    print("Extracting features with VGG16...")
    features = base_model.predict(images)
    
    # Flatten features: (Num_Images, 7, 7, 512) -> (Num_Images, 25088)
    # This might be too large for GRU input depending on RAM.
    # We can use GlobalAveragePooling in VGG16 or just flatten.
    # Let's try flattening for now, if it's too big, we'll pool.
    features_flat = features.reshape(features.shape[0], -1)
    
    return features_flat, labels

def extract_frames_from_video(video_path, output_folder):
    """
    Extracts frames from a video file and saves them as images.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    count = 0
    frameRate = cap.get(5) # frame rate
    
    print(f"Extracting frames from {video_path}...")
    while(cap.isOpened()):
        frameId = cap.get(1) # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        # Extract frames at the same rate as the original logic (roughly 1 fps if encoded as such, or every second)
        # The original code used: if (frameId % math.floor(frameRate) == 0):
        # We need to replicate this behavior to match mapping.csv
        if (frameId % np.floor(frameRate) == 0):
            filename = os.path.join(output_folder, f"{count}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
    cap.release()
    print("Frame extraction detailed.")

